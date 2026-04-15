import streamlit as st
import requests
import hashlib
import hmac
import json
import random
import matplotlib.pyplot as plt
import time
import math
import os
from sqlalchemy import select

import pandas as pd

# DB helpers
from database import (
    Prompt,
    User,
    Feedback,
    SessionLocal,
    register_user,
    authenticate_user,
    change_password,
    add_feedback,
    get_top_topics,
    get_top_styles,
    get_feedback_counts_by_model,
    get_active_users,
    record_choice,
    get_choice_dataset
)

# ranker utilities
from ranker import train_basic, train_with_embeddings, compare_models, load_ranker, predict_with_ranker

# ==== CONFIG/SECRETS ====
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
# Allow specifying which Gemini model to use via secrets (e.g. "gemini-1.5-flash" or "gemini-2.0-flash")
# Default to the commonly used 1.5 flash if not set to avoid unexpected 2.0 calls.
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
# Rate limit (seconds) to wait before calling the API. Can be set in secrets as a number.
try:
    _rl = st.secrets.get("GEMINI_RATE_LIMIT_SECONDS", None)
    GEMINI_RATE_LIMIT_SECONDS = float(_rl) if _rl is not None else 1.0
except Exception:
    GEMINI_RATE_LIMIT_SECONDS = 1.0
try:
    _gt = st.secrets.get("GEMINI_REQUEST_TIMEOUT", None)
    GEMINI_REQUEST_TIMEOUT = float(_gt) if _gt is not None else 30.0
except Exception:
    GEMINI_REQUEST_TIMEOUT = 30.0

ADMIN_USERNAME = st.secrets.get("ADMIN_USERNAME", None)
ADMIN_PW_SALT = st.secrets.get("ADMIN_PW_SALT", None)
ADMIN_PW_HASH = st.secrets.get("ADMIN_PW_HASH", None)

RANKER_PATH = "ranker.pkl"

# helpers
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass


def _clear_generation_state():
    """Clear any previously displayed outputs or hybrid state when inputs change."""
    st.session_state["show_output"] = False
    st.session_state["generated_text"] = None
    st.session_state["generated_prompt_id"] = None
    st.session_state["awaiting_hybrid_choice"] = False
    st.session_state["hybrid_offline_text"] = None
    st.session_state["hybrid_gemini_text"] = None
    st.session_state["hybrid_offline_id"] = None
    st.session_state["hybrid_gemini_id"] = None
    st.session_state["hybrid_choice_model"] = None

def verify_admin_password(plain_password: str) -> bool:
    if not (ADMIN_PW_SALT and ADMIN_PW_HASH):
        return False
    computed = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode(),
        ADMIN_PW_SALT.encode(),
        200_000
    ).hex()
    return hmac.compare_digest(computed, ADMIN_PW_HASH)


# prompt generators
def generate_template_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    if tool.lower() == "gamma":
        return (
            f"Create a {style} {content_type} about {topic}. "
            "Include engaging visuals, clear text, and a professional layout."
        )

    elif tool.lower() == "canva":
        target = platform if platform else "social media"
        sentence = f"Design a {style} {content_type} about {topic}, tailored for {target}. "

        if color_palette:
            sentence += f"Use a {color_palette} color scheme"
            if mood:
                sentence += f" to create a {mood} atmosphere. "
            else:
                sentence += ". "
        elif mood:
            sentence += f"Aim for a {mood} feel in the overall design. "

        sentence += "Make it engaging and visually appealing."
        return sentence

    return f"Create a {style} {content_type} about {topic}."


def _log_gemini_error(error_type, attempt, **extra):
    """Append a structured JSON line to the Gemini error log."""
    import datetime
    try:
        os.makedirs("artifacts", exist_ok=True)
        entry = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "type": error_type,
            "attempt": attempt,
            **extra,
        }
        with open(os.path.join("artifacts", "gemini_errors.log"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# Maximum retries and exponential backoff base for Gemini calls
_GEMINI_MAX_RETRIES = 3
_GEMINI_BACKOFF_BASE = 1  # seconds — gives 1s, 2s, 4s


def generate_gemini_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    user_parts = [f"I want to create a {style} {content_type} about {topic}."]
    if platform:
        user_parts.append(f"Tailored for {platform}.")
    if color_palette and mood:
        user_parts.append(f"Use a {color_palette} color scheme to create a {mood} atmosphere.")
    elif color_palette:
        user_parts.append(f"Use a {color_palette} color scheme.")
    elif mood:
        user_parts.append(f"Create a {mood} atmosphere.")

    user_parts.append(f"Write the prompt as if the user will paste it into {tool}.")
    user_msg = " ".join([p for p in user_parts if p])

    # Build request URL based on configured GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": user_msg}]}]}

    def _fallback(reason):
        """Return the offline template with a prefixed warning."""
        tpl = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
        return f"⚠️ {reason}; showing offline template instead:\n\n" + tpl

    last_exception = None

    for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
        # Exponential backoff before retries: 1s, 2s, 4s
        if attempt > 1:
            backoff = _GEMINI_BACKOFF_BASE * (2 ** (attempt - 2))
            time.sleep(backoff)

        # Rate-limit pause before every request
        time.sleep(GEMINI_RATE_LIMIT_SECONDS)

        # ---- Network-level errors ----
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_REQUEST_TIMEOUT)
        except requests.exceptions.Timeout as e:
            last_exception = e
            _log_gemini_error("timeout", attempt, timeout=GEMINI_REQUEST_TIMEOUT)
            if attempt >= _GEMINI_MAX_RETRIES:
                return _fallback(f"Gemini request timed out after {GEMINI_REQUEST_TIMEOUT}s (attempts={attempt})")
            continue
        except requests.exceptions.RequestException as e:
            last_exception = e
            _log_gemini_error("request_exception", attempt, error=str(e))
            if attempt >= _GEMINI_MAX_RETRIES:
                return _fallback(f"Gemini request failed ({e})")
            continue

        # ---- Parse response body ----
        try:
            data = resp.json()
        except Exception:
            data = resp.text

        # ---- Transient server errors (502/503/504) — retry ----
        if resp.status_code in (502, 503, 504):
            last_exception = Exception(f"HTTP {resp.status_code}")
            _log_gemini_error("http_5xx", attempt, status=resp.status_code,
                              body_preview=resp.text[:400])
            if attempt >= _GEMINI_MAX_RETRIES:
                return _fallback(f"Gemini returned HTTP {resp.status_code} after {attempt} attempts")
            continue

        # ---- Rate-limit (429) — non-retriable, immediate fallback ----
        if resp.status_code == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") == 429):
            _log_gemini_error("http_429", attempt, status=429,
                              body_preview=resp.text[:400])
            detail = data.get("error", {}).get("message") if isinstance(data, dict) else None
            suffix = f": {detail}" if detail else ""
            return _fallback(f"Gemini free-tier quota exceeded (HTTP 429){suffix}")

        # ---- Other HTTP errors — non-retriable ----
        if not resp.ok:
            _log_gemini_error("http_error", attempt, status=resp.status_code,
                              body_preview=resp.text[:400])
            return f"Gemini API error: HTTP {resp.status_code} - {data}"

        # ---- Success ----
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"Unexpected response format: {data}"

    # Should not reach here, but safety fallback
    return _fallback(f"Gemini call exhausted {_GEMINI_MAX_RETRIES} retries (last error: {last_exception})")


def test_gemini_key_once(timeout_override: float | None = None):
    """Perform a single, lightweight GET to the configured model endpoint to verify the key and model.

    Returns a tuple: (ok: bool, message: str, status_code: int|None)
    """
    if not GEMINI_API_KEY:
        return False, "Gemini API key not configured.", None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
    headers = {"X-goog-api-key": GEMINI_API_KEY}
    timeout = float(timeout_override) if timeout_override is not None else min(GEMINI_REQUEST_TIMEOUT, 10.0)

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return False, f"Request timed out after {timeout}s.", None
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}", None

    # Try to parse body safely
    try:
        body = resp.json()
    except Exception:
        body = resp.text

    if resp.status_code == 200:
        # Model reachable and readable
        return True, f"Success: model '{GEMINI_MODEL}' is reachable (HTTP 200).", 200
    if resp.status_code == 404:
        # Model not found
        return False, f"Model '{GEMINI_MODEL}' not found (HTTP 404). Consider using a supported model (e.g., 'gemini-2.5-flash').", 404
    if resp.status_code == 401 or resp.status_code == 403:
        return False, f"Authentication/permission error (HTTP {resp.status_code}). Check the API key and project permissions.", resp.status_code
    if resp.status_code == 429:
        # Quota exceeded
        detail = None
        if isinstance(body, dict):
            detail = body.get("error", {}).get("message")
        msg = f"Quota exceeded (HTTP 429). {detail or ''}".strip()
        return False, msg, 429

    # Fallback: return status and a short body preview
    summary = body if isinstance(body, str) else str(body)[:200]
    return False, f"Unexpected HTTP {resp.status_code}: {summary}", resp.status_code



# DB operations for prompts (save/delete)
def save_prompt(tool, content_type, topic, style, generated_prompt, model_used="offline", user_id=None,
                platform_name=None, color_palette=None, mood=None, used_hybrid=False):
    with SessionLocal() as session:
        new_prompt = Prompt(
            tool=tool,
            content_type=content_type,
            topic=topic,
            style=style,
            platform_name=platform_name,
            color_palette=color_palette,
            mood=mood,
            generated_text=generated_prompt,
            model_used=model_used,
            used_hybrid=bool(used_hybrid),
            user_id=user_id
        )
        session.add(new_prompt)
        session.commit()
        session.refresh(new_prompt)
        return new_prompt

def delete_prompt(prompt_id):
    with SessionLocal() as session:
        prompt = session.get(Prompt, prompt_id)
        if prompt:
            session.delete(prompt)
            session.commit()
            return True
        return False

def handle_delete(prompt_id):
    if delete_prompt(prompt_id):
        st.success("Prompt deleted.")
    else:
        st.error("Prompt not found.")


# session_state defaults
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None
if "last_search_tool" not in st.session_state:
    st.session_state["last_search_tool"] = "All"
if "last_search_topic" not in st.session_state:
    st.session_state["last_search_topic"] = ""
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# generation/selection state
for key, default in [
    ("show_output", False),
    ("generated_text", None),
    ("generated_prompt_id", None),
    ("awaiting_hybrid_choice", False),
    ("hybrid_offline_text", None),
    ("hybrid_gemini_text", None),
    ("hybrid_offline_id", None),
    ("hybrid_gemini_id", None),
    ("hybrid_choice_model", None),
    ("ensemble_log", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# admin UI
def admin_login_ui():
    st.subheader("🔐 Admin Login")
    with st.form("admin_login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if ADMIN_USERNAME is None or ADMIN_PW_SALT is None or ADMIN_PW_HASH is None:
            st.error("Admin credentials are not configured. Add them to Streamlit secrets.")
            return False
        if username != ADMIN_USERNAME:
            st.error("Invalid username.")
            return False
        if verify_admin_password(password):
            st.session_state["admin_logged_in"] = True
            st.success("✅ Admin login successful.")
            return True
        else:
            st.error("Invalid password.")
            return False
    return False


# Admin: Dashboard & Ranker controls
def show_dashboard():
    st.header("📊 Dashboard")
    st.markdown("Basic analytics: top topics/styles, feedback per model, active users.")

    top_topics = get_top_topics(10)
    if top_topics:
        topics = [t for t, c in top_topics]
        counts = [c for t, c in top_topics]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(topics, counts)
        ax.set_title("Top Topics")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No prompt data for topics yet.")

    top_styles = get_top_styles(10)
    if top_styles:
        styles = [s or "(empty)" for s, c in top_styles]
        counts = [c for s, c in top_styles]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(styles, counts)
        ax.set_title("Top Styles")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No prompt data for styles yet.")

    # feedback_by_model = get_feedback_counts_by_model()
    # if feedback_by_model:
    #     models = []
    #     pos = []
    #     neg = []
    #     for model, stats in feedback_by_model.items():
    #         models.append(model)
    #         pos.append(stats.get("positive", 0))
    #         neg.append(stats.get("negative", 0))
    #     x = range(len(models))
    #     fig, ax = plt.subplots(figsize=(6, 3))
    #     width = 0.35
    #     ax.bar([i - width/2 for i in x], pos, width=width)
    #     ax.bar([i + width/2 for i in x], neg, width=width)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(models, rotation=45)
    #     ax.legend(["Positive", "Negative"])
    #     st.pyplot(fig)
    # else:
    #     st.info("No feedback data yet.")

    active = get_active_users(20)
    if active:
        df = pd.DataFrame([{"username": u, "prompt_count": c} for u, c in active])
        st.subheader("Active users (by prompts created)")
        st.dataframe(df)
    else:
        st.info("No user activity yet.")

    # Gemini API health test (admin-only)
    st.markdown("---")
    st.subheader("🔎 Gemini API Health Check")
    st.caption(f"Configured model: **{GEMINI_MODEL}** — rate-limit wait: **{GEMINI_RATE_LIMIT_SECONDS}s**")
    # Model can be configured via `GEMINI_MODEL` secret (default: gemini-2.5-flash)

    if GEMINI_API_KEY:
        try:
            masked = f"***{GEMINI_API_KEY[-6:]}"
        except Exception:
            masked = "(configured)"
        st.info(f"Gemini key configured (last 6 chars: {masked}). Click 'Test' to check status.")
        st.caption("Performs a single lightweight model GET to verify key/model. No model listing will be displayed.")
        if st.button("Test Gemini key"):
            with st.spinner("Testing Gemini key..."):
                ok, msg, status = test_gemini_key_once()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        # Show recent Gemini errors (if any)
        with st.expander("Recent Gemini errors (last 10)", expanded=False):
            try:
                log_path = os.path.join("artifacts", "gemini_errors.log")
                if os.path.exists(log_path):
                    lines = open(log_path, "r", encoding="utf-8").read().splitlines()
                    for ln in lines[-10:][::-1]:
                        try:
                            obj = json.loads(ln)
                            ts = obj.get("ts", "?")
                            typ = obj.get("type", "")
                            status = obj.get("status")
                            body = obj.get("body_preview") or obj.get("error") or ""
                            st.markdown(f"- **{ts}** — `{typ}` {status or ''} — {body}")
                        except Exception:
                            st.markdown(f"- {ln[:200]}")
                else:
                    st.info("No Gemini errors logged yet.")
            except Exception:
                st.info("Could not read log file.")
    else:
        st.info("Gemini API key not configured in Streamlit secrets.")

    # Keep Dashboard minimal per request — no model listing/diagnostics UI


def admin_panel():
    st.title("🛠️ Admin Panel")
    menu = ["Dashboard", "Train Ranker", "SHAP Explanations", "View Users", "View Prompts", "Choices Dataset"]
    choice = st.selectbox("Choose view", menu, key="admin_menu_select")

    if choice == "Dashboard":
        show_dashboard()
        return

    if choice == "SHAP Explanations":
        st.header("🔍 SHAP Explanations")
        st.markdown("Interpretable explanations for neural ranker predictions using SHAP.")

        tab_global, tab_local = st.tabs(["📊 Global Importance", "🔬 Local Explanation"])

        # ---- Global Tab ----
        with tab_global:
            st.subheader("Global Feature Importance")
            st.markdown(
                "Compute SHAP values across saved prompts to identify which "
                "embedding dimensions the neural ranker relies on most."
            )

            max_samples = st.number_input(
                "Max samples", min_value=10, max_value=500, value=100,
                key="shap_global_max",
            )

            # Check for cached results

            cached_global = None
            global_path = os.path.join("results", "shap_global.json")
            if os.path.exists(global_path):
                try:
                    with open(global_path, "r", encoding="utf-8") as _f:
                        cached_global = json.load(_f)
                except Exception:
                    pass

            if cached_global:
                st.info(
                    f"Showing cached results ({cached_global.get('n_samples', '?')} samples, "
                    f"{cached_global.get('explainer', '?')}). Click below to recompute."
                )
                try:
                    from shap_explain import plot_global_importance
                    fig = plot_global_importance(cached_global)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error plotting: {e}")

            if st.button("Compute Global SHAP", key="btn_global_shap"):
                with st.spinner("Computing global SHAP values (this may take a minute)..."):
                    try:
                        # Collect texts from DB
                        with SessionLocal() as session:
                            all_prompts = session.execute(select(Prompt)).scalars().all()
                            texts = [p.generated_text for p in all_prompts if p.generated_text]

                        if len(texts) < 5:
                            st.error("Need at least 5 saved prompts. Generate more prompts first.")
                        else:
                            from shap_explain import compute_global_shap, plot_global_importance
                            result = compute_global_shap(
                                texts, max_samples=int(max_samples),
                            )
                            st.success(
                                f"SHAP computed over {result['n_samples']} samples "
                                f"using {result['explainer']}. Saved to results/shap_global.json."
                            )
                            fig = plot_global_importance(result)
                            st.pyplot(fig)
                            plt.close(fig)

                            with st.expander("Raw top dimensions", expanded=False):
                                st.json(result["top_dims"])
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")

        # ---- Local Tab ----
        with tab_local:
            st.subheader("Local Prediction Explanation")
            st.markdown(
                "Enter a prompt text to see the neural ranker's prediction "
                "and which embedding dimensions contributed most."
            )

            local_text = st.text_area(
                "Prompt text to explain:", height=100, key="shap_local_input",
            )

            if st.button("Explain Prediction", key="btn_local_shap"):
                if not local_text or not local_text.strip():
                    st.error("Enter a prompt text to explain.")
                else:
                    with st.spinner("Computing local SHAP explanation..."):
                        try:
                            # Gather background texts from DB
                            with SessionLocal() as session:
                                bg_prompts = session.execute(select(Prompt)).scalars().all()
                                bg_texts = [p.generated_text for p in bg_prompts if p.generated_text][:50]

                            from shap_explain import compute_local_shap, plot_local_explanation
                            result = compute_local_shap(
                                local_text.strip(),
                                background_texts=bg_texts if bg_texts else None,
                            )
                            pred = result["predictions"][0]

                            col_label, col_prob = st.columns(2)
                            col_label.metric("Predicted Label", pred["predicted_label"])
                            col_prob.metric("Probability", f"{pred['probability']:.4f}")

                            fig = plot_local_explanation(pred)
                            st.pyplot(fig)
                            plt.close(fig)

                            with st.expander("Top contributing dimensions", expanded=False):
                                st.json(pred["top_contributions"])

                            st.caption(f"Explainer: {result['explainer']} | Background: {result['n_background']} samples")
                        except Exception as e:
                            st.error(f"SHAP explanation failed: {e}")

            # Show cached local result if it exists
            local_path = os.path.join("results", "shap_local.json")
            if os.path.exists(local_path):
                with st.expander("Last saved local explanation (shap_local.json)", expanded=False):
                    try:
                        with open(local_path, "r", encoding="utf-8") as _f:
                            st.json(json.load(_f))
                    except Exception:
                        st.info("Could not load cached results.")

        return

    if choice == "Train Ranker":
        st.header("Train / Retrain Ranker")
        st.markdown("Train the selector/ranker using recorded choices (preferred) or positive feedback.")

        emb_models = [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-MiniLM-L12-v2"
        ]
        selected_emb = st.selectbox("Embedding model", emb_models, index=0, key="emb_model_select")

        # Data sufficiency indicator: show number of examples and samples per class
        dataset = get_choice_dataset()
        if not dataset:
            st.warning("No choice dataset available yet — generate Hybrid prompts and record choices to collect labeled examples.")
            samples_info = {}
        else:
            # dataset is list of (text, label)
            _, labels = zip(*dataset)
            from collections import Counter
            counts = Counter(labels)
            total = len(labels)
            samples_info = dict(counts)
            st.info(f"Dataset: {total} examples across {len(counts)} classe(s). Samples per class: {dict(counts)}")

        # Calculate CV feasibility for default 5-fold
        def _cv_feasible(counts_dict, k=5):
            if not counts_dict:
                return False
            return all(c >= k for c in counts_dict.values())

        if samples_info and not _cv_feasible(samples_info, k=5):
            st.info("tfidf cross-validation not feasible on this dataset (too few samples per class).")

        # Embedding model quick test button
        if st.button("Test embedding model"):
            with st.spinner("Testing embedding model load..."):
                try:
                    from ranker import _load_embedder
                    emb = _load_embedder(selected_emb)
                    # if it's a SentenceTransformer instance, try a short encode
                    if hasattr(emb, "encode"):
                        _ = emb.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
                    else:
                        _ = emb(["test"])
                    st.success(f"Embedding model '{selected_emb}' loaded successfully.")
                except Exception as e:
                    st.error("Embedding model test failed: " + str(e))

        st.markdown("---")
        st.subheader("Bulk labeling helper")
        st.markdown("Quickly create many hybrid labeling tasks from a list of topics so you can label them in-app or have users label them.")
        topics_text = st.text_area("Topics (one per line)", key="bulk_topics")
        variations = st.number_input("Variations per topic", min_value=1, max_value=50, value=5, key="bulk_vars")
        bulk_tool = st.selectbox("Tool for prompts", ["Gamma", "Canva"], key="bulk_tool")
        bulk_style = st.text_input("Style for bulk prompts (e.g. modern)", key="bulk_style")
        bulk_platform = st.text_input("Platform (optional)", key="bulk_platform")
        bulk_colors = st.text_input("Color palette (optional)", key="bulk_colors")
        bulk_mood = st.text_input("Mood (optional)", key="bulk_mood")

        if st.button("Create hybrid labeling tasks"):
            lines = [l.strip() for l in (topics_text or "").splitlines() if l.strip()]
            if not lines:
                st.error("Enter at least one topic (one per line).")
            else:
                created = 0
                import datetime
                os.makedirs("artifacts", exist_ok=True)
                pairs_path = os.path.join("artifacts", "hybrid_pairs.jsonl")
                for topic in lines:
                    for i in range(int(variations)):
                        # create minor variation by appending index
                        t = f"{topic} ({i+1})" if int(variations) > 1 else topic
                        offline_text = generate_template_prompt(bulk_tool, "presentation", t, bulk_style or "modern", bulk_platform or None, bulk_colors or None, bulk_mood or None)
                        gemini_text = generate_gemini_prompt(bulk_tool, "presentation", t, bulk_style or "modern", bulk_platform or None, bulk_colors or None, bulk_mood or None)
                        # Save prompts to DB
                        offline_obj = save_prompt(bulk_tool, "presentation", t, bulk_style or "modern", offline_text, model_used="offline", user_id=None, platform_name=bulk_platform or None, color_palette=bulk_colors or None, mood=bulk_mood or None, used_hybrid=True)
                        gemini_obj = save_prompt(bulk_tool, "presentation", t, bulk_style or "modern", gemini_text, model_used="gemini", user_id=None, platform_name=bulk_platform or None, color_palette=bulk_colors or None, mood=bulk_mood or None, used_hybrid=True)
                        # record pair in artifacts file
                        entry = {"offline_id": offline_obj.id, "gemini_id": gemini_obj.id, "created": datetime.datetime.utcnow().isoformat() + "Z", "labeled": False}
                        with open(pairs_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(entry) + "\n")
                        created += 1
                st.success(f"Created {created} hybrid labeling tasks and saved pairs to {pairs_path}.")

        st.markdown("---")
        st.subheader("Pending hybrid labeling tasks")
        st.markdown("Use this interface to label tasks created above. Select a user to attribute the choices to (required to build training dataset).")
        # fetch users for attribution
        with SessionLocal() as session:
            users = session.execute(select(User)).scalars().all()
            user_options = [(u.id, u.username) for u in users]
        if user_options:
            selected_user = st.selectbox("Record choices as user:", options=[(uid, name) for uid, name in user_options], format_func=lambda t: t[1], key="label_user_select")
            selected_user_id = selected_user[0]
        else:
            selected_user_id = None
            st.warning("No users exist to attribute choices to. Create users first or attribute choices to an existing user.")

        # read pending pairs

        pairs_path = os.path.join("artifacts", "hybrid_pairs.jsonl")
        pending = []
        if os.path.exists(pairs_path):
            for ln in open(pairs_path, "r", encoding="utf-8").read().splitlines():
                try:
                    obj = json.loads(ln)
                    if not obj.get("labeled"):
                        pending.append(obj)
                except Exception:
                    continue

        if not pending:
            st.info("No pending hybrid pairs. Create some with the Bulk labeling helper above.")
        else:
            # show first 10 pending
            for idx, p in enumerate(pending[:10]):
                off_id = p.get("offline_id")
                gem_id = p.get("gemini_id")
                with SessionLocal() as session:
                    off = session.get(Prompt, off_id)
                    gem = session.get(Prompt, gem_id)
                if not off or not gem:
                    st.warning(f"Pair {idx+1}: prompts not found (maybe deleted)")
                    continue
                st.markdown(f"**Pair #{idx+1}** — Topic: {off.topic}")
                st.markdown("**Offline variant:**")
                st.code(off.generated_text, language="markdown")
                st.markdown("**Gemini variant:**")
                st.code(gem.generated_text, language="markdown")
                col_l, col_r = st.columns([1,1])
                if col_l.button("Choose Offline", key=f"choose_off_{idx}"):
                    if selected_user_id is None:
                        st.error("Select a user to attribute the choice to before labeling.")
                    else:
                        ok, res = record_choice(selected_user_id, off_id, gem_id, off_id, "offline")
                        if ok:
                            # mark labeled
                            _mark_pair_labeled(pairs_path, off_id, gem_id, off_id, "offline", selected_user_id)
                            st.success("Choice recorded (offline).")
                        else:
                            st.error("Could not record choice: " + (res or "unknown"))
                if col_r.button("Choose Gemini", key=f"choose_gem_{idx}"):
                    if selected_user_id is None:
                        st.error("Select a user to attribute the choice to before labeling.")
                    else:
                        ok, res = record_choice(selected_user_id, off_id, gem_id, gem_id, "gemini")
                        if ok:
                            _mark_pair_labeled(pairs_path, off_id, gem_id, gem_id, "gemini", selected_user_id)
                            st.success("Choice recorded (gemini).")
                        else:
                            st.error("Could not record choice: " + (res or "unknown"))

        if st.button("Train embedding-based ranker (if available)"):
            # It will call ranker.train_with_embeddings using dataset from get_choice_dataset
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available. Generate both variants and choose a variant in the app to collect data.")
            else:
                texts, labels = zip(*dataset)
                try:
                    acc, rep = train_with_embeddings(list(texts), list(labels), embed_model_name=selected_emb, save_path=RANKER_PATH)
                    st.success(f"Embedding-based ranker trained (cv acc ≈ {acc:.3f})")
                    st.json(rep)
                except Exception as e:
                    # Log the error for later inspection and surface a friendly message
                    try:
                        import datetime
                        os.makedirs("artifacts", exist_ok=True)
                        with open("artifacts/ranker_errors.log", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"ts": datetime.datetime.utcnow().isoformat() + "Z", "error": str(e)}) + "\n")
                    except Exception:
                        pass
                    st.error("Embedding train failed: " + str(e))


        if st.button("Train TF-IDF ranker (fallback)"):
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available.")
            else:
                texts, labels = zip(*dataset)
                try:
                    acc, rep = train_basic(list(texts), list(labels), save_path=RANKER_PATH)
                    st.success(f"TF-IDF ranker trained (cv acc ≈ {acc:.3f})")
                    st.json(rep)
                    if rep.get("note") == "no_cv":
                        st.info("TF-IDF cross-validation not feasible on this dataset (too few samples per class). The model was still trained but CV was skipped.")
                except Exception as e:
                    st.error("TF-IDF train failed: " + str(e))

        st.markdown("---")
        st.subheader("Model comparison (5-fold CV)")
        metric_options = [
            "accuracy",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
        ]
        selected_metric = st.selectbox("Primary metric to display", metric_options, index=0, key="model_comp_metric")

        if st.button("Run quick comparison"):
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available.")
            else:
                texts, labels = zip(*dataset)
                try:
                    results = compare_models(list(texts), list(labels), embed_model_name=selected_emb, cv=5)
                    st.json(results)

                    # Simple bar viz of mean accuracies (skip models that errored)
                    labels_x = []
                    means = []
                    for k, v in results.items():
                        if isinstance(v, dict) and v.get("error"):
                            st.warning(f"{k} comparison unavailable: {v.get('error')}")
                            continue
                        # Respect new report shape: prefer 'means' dict, fallback to legacy 'mean'
                        mean_val = None
                        if isinstance(v, dict):
                            mean_val = v.get("means", {}).get(selected_metric)
                            if mean_val is None:
                                # fallback for legacy or where metric not computed
                                mean_val = v.get("mean")
                        try:
                            mean_num = float(mean_val) if mean_val is not None and not (isinstance(mean_val, float) and math.isnan(mean_val)) else None
                        except Exception:
                            mean_num = None
                        if mean_num is None:
                            st.info(f"{k} cross-validation not feasible on this dataset (too few samples per class).")
                            continue
                        labels_x.append(k)
                        means.append(mean_num)

                    if labels_x and means:
                        fig, ax = plt.subplots(figsize=(4,3))
                        ax.bar(labels_x, means)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel(f"Mean CV {selected_metric}")
                        st.pyplot(fig)
                        plt.close(fig)

                    # Build a per-model metric breakdown table
                    try:
                        rows = []
                        for k, v in results.items():
                            row = {"model": k}
                            if isinstance(v, dict) and v.get("error"):
                                row["error"] = v.get("error")
                                rows.append(row)
                                continue
                            for m in metric_options:
                                mean_val = v.get("means", {}).get(m)
                                folds = v.get("cv_scores", {}).get(m, [])
                                row[f"{m}_mean"] = mean_val
                                # store fold values as JSON string for readability
                                row[f"{m}_folds"] = json.dumps(folds)
                            rows.append(row)
                        if rows:
                            df = pd.DataFrame(rows).set_index("model")
                            with st.expander("Per-model metric breakdown"):
                                st.dataframe(df)
                    except Exception:
                        # non-fatal UI rendering failure
                        pass
                except Exception as e:
                    st.error("Comparison failed: " + str(e))
        return

    # Other admin views
    with SessionLocal() as session:
        if choice == "View Users":
            st.subheader("👤 Users")
            users = session.execute(select(User)).scalars().all()
            if users:
                df = pd.DataFrame([{
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "created_at": u.created_at
                } for u in users])
                st.dataframe(df)
            else:
                st.info("No users found.")

        elif choice == "View Prompts":
            st.subheader("📝 Prompts")
            prompts = session.execute(select(Prompt)).scalars().all()
            if prompts:
                for p in prompts:
                    st.markdown(f"**ID:** {p.id} | **Tool:** {p.tool} | **Topic:** {p.topic} | **Model:** {p.model_used} | **HybridUsed:** {p.used_hybrid} | **User ID:** {p.user_id}")
                    st.code(p.generated_text, language="markdown")
                    col_d, col_sp = st.columns([0.6, 8])
                    with col_d:
                        st.button("🗑️ Delete", key=f"admin_delete_{p.id}", on_click=handle_delete, args=(p.id,))
                    st.markdown("---")
            else:
                st.info("No prompts found.")

        # elif choice == "View Feedback":
        #     st.subheader("💬 Feedback")
        #     feedbacks = session.execute(select(Feedback)).scalars().all()
        #     if feedbacks:
        #         df = pd.DataFrame([{
        #             "id": f.id,
        #             "rating": f.rating,
        #             "comments": f.comments,
        #             "timestamp": f.timestamp,
        #             "user_id": f.user_id,
        #             "prompt_id": f.prompt_id
        #         } for f in feedbacks])
        #         st.dataframe(df)
        #     else:
        #         st.info("No feedback found.")

        elif choice == "Choices Dataset":
            st.subheader("🎯 Choices dataset (for ranker)")
            rows = get_choice_dataset()
            if rows:
                st.info(f"{len(rows)} choice examples available.")
                df = pd.DataFrame([{"text": t, "label": l} for t, l in rows])
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="choices_dataset.csv", mime="text/csv")
            else:
                st.info("No choices recorded yet. Use Generate → Hybrid → Generate both and choose to record choices.")

def _mark_pair_labeled(pairs_path: str, offline_id: int, gemini_id: int, chosen_id: int, chosen_model: str, user_id: int):
    # read all, update matching pair

    lines = []
    try:
        with open(pairs_path, "r", encoding="utf-8") as fh:
            for ln in fh.read().splitlines():
                try:
                    obj = json.loads(ln)
                    if obj.get("offline_id") == offline_id and obj.get("gemini_id") == gemini_id and not obj.get("labeled"):
                        obj["labeled"] = True
                        obj["chosen_id"] = chosen_id
                        obj["chosen_model"] = chosen_model
                        obj["labeled_by"] = int(user_id)
                    lines.append(json.dumps(obj))
                except Exception:
                    lines.append(ln)
        with open(pairs_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    except Exception:
        pass
    


# User auth UI
def user_auth_ui():
    if st.session_state.get("user"):
        return True

    st.subheader("🔑 User Login / Signup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Login**")
        login_username = st.text_input("Username (login)", key="login_username")
        login_password = st.text_input("Password (login)", type="password", key="login_password")
        if st.button("Login as user", key="login_btn"):
            user = authenticate_user(login_username, login_password)
            if user:
                if not user.email:
                    st.error("Email required for login; user record missing email.")
                else:
                    st.session_state["user"] = {"id": user.id, "username": user.username, "role": getattr(user, "role", None)}
                    st.success(f"Welcome back, {user.username}!")
                    safe_rerun()
            else:
                st.error("Invalid username or password.")

    with col2:
        st.markdown("**Signup**")
        signup_username = st.text_input("Choose username", key="signup_username")
        signup_password = st.text_input("Choose password", type="password", key="signup_password")
        signup_email = st.text_input("Email (required)", key="signup_email")
        if st.button("Create account", key="signup_btn"):
            if not signup_username or not signup_password or not signup_email:
                st.error("Enter username, password, and email (email is mandatory).")
            else:
                success, payload = register_user(signup_username, signup_password, signup_email)
                if not success:
                    if payload == "username_exists":
                        st.error("Username already exists — choose another.")
                    elif payload == "email_exists":
                        st.error("Email already in use.")
                    elif payload == "email_required":
                        st.error("Email required.")
                    else:
                        st.error("Could not create account (db error).")
                else:
                    st.success("Account created. You can now log in.")
    return False

def account_sidebar():
    if not st.session_state.get("user"):
        return

    st.sidebar.markdown(f"👋 Logged in as **{st.session_state['user']['username']}**")
    st.sidebar.markdown("### Account")
    with st.sidebar.expander("🔒 Change password", expanded=False):
        old_pw = st.text_input("Current password", type="password", key="cp_old")
        new_pw = st.text_input("New password", type="password", key="cp_new")
        confirm_pw = st.text_input("Confirm new password", type="password", key="cp_confirm")
        if st.button("Change password", key="cp_btn"):
            if not old_pw or not new_pw:
                st.error("Enter both current and new password.")
            elif new_pw != confirm_pw:
                st.error("New password and confirm do not match.")
            else:
                username = st.session_state["user"]["username"]
                ok, msg = change_password(username, old_pw, new_pw)
                if ok:
                    st.success("Password changed successfully.")
                else:
                    if msg == "incorrect_old_password":
                        st.error("Current password is incorrect.")
                    else:
                        st.error("Could not change password. Try again.")

    if st.sidebar.button("🚪 Logout"):
        st.session_state["user"] = None
        st.success("Logged out successfully.")
        safe_rerun()


# Main app: prompt generation + history with choice recording
def run_prompt_generator():
    st.title("✨ Prompt Generator for AI Tools")
    st.markdown("Generate perfect prompts for **Gamma** and **Canva** with optional Gemini AI support!")

    st.sidebar.header("🔍 Examples")
    st.sidebar.markdown("- Gamma: Modern sales deck with key metrics and customer testimonials")
    st.sidebar.markdown("- Gamma: Educational presentation on renewable energy trends")
    st.sidebar.markdown("- Canva: Instagram story announcing a product launch")
    st.sidebar.markdown("- Canva: Facebook post graphic for a holiday sale")
    st.sidebar.markdown("---")
    account_sidebar()
    st.sidebar.markdown("👨‍💻 Made with ❤️ by [Parth Shinge](https://github.com/parth-shinge)")

    def _on_global_change():
        _clear_generation_state()
    model_choice = st.selectbox("Model choice:", ["Offline", "Gemini", "Hybrid"], key="model_choice_select", on_change=_on_global_change)
    tool = st.selectbox("Choose a tool:", ["Gamma", "Canva"], key="tool_select", on_change=_on_global_change)

    # ── Reset form inputs when the selected tool changes ──
    if "last_tool" not in st.session_state:
        st.session_state.last_tool = tool
    if st.session_state.last_tool != tool:
        _form_input_keys = [
            "content_type_input",
            "topic_input",
            "style_input",
            "canva_platform",
            "canva_colors",
            "canva_mood",
            "gen_both_checkbox",
        ]
        for key in _form_input_keys:
            if key in st.session_state:
                del st.session_state[key]
        _clear_generation_state()
        st.session_state.last_tool = tool
        st.rerun()

    tab1, tab2 = st.tabs(["✨ Generate Prompt", "📂 Search History"])

    # Generate tab
    with tab1:
        if not user_auth_ui():
            st.info("Please login or sign up to generate and save prompts.")
            return

        st.subheader("✨ Generate a New Prompt")
        with st.form("generate_form", clear_on_submit=False):
            content_type = st.text_input("Content type (e.g. presentation, infographic, poster):", key="content_type_input")
            topic = st.text_input("Topic:", key="topic_input")
            style = st.text_input("Style (e.g. modern, playful, minimalist):", key="style_input")

            # canva extras
            platform = color_palette = mood = None
            if st.session_state.get("tool_select") == "Canva":
                platform = st.text_input("Platform (e.g. Instagram, Facebook):", key="canva_platform")
                color_palette = st.text_input("Color palette (e.g. bright, pastel, dark):", key="canva_colors")
                mood = st.text_input("Mood (e.g. energetic, calm, elegant):", key="canva_mood")

            if model_choice in ["Offline", "Gemini"]:
                gen_both_and_choose = st.checkbox("Generate both variants and choose (Hybrid mode)", value=False, key="gen_both_checkbox")
            else: 
                gen_both_and_choose = False

            submitted = st.form_submit_button("Generate Prompt")

        if submitted:
            # Clear outputs when inputs changed compared to last submission
            signature_parts = [
                st.session_state.get("content_type_input") or "",
                st.session_state.get("topic_input") or "",
                st.session_state.get("style_input") or "",
                st.session_state.get("canva_platform") or "",
                st.session_state.get("canva_colors") or "",
                st.session_state.get("canva_mood") or "",
                st.session_state.get("model_choice_select") or "",
                st.session_state.get("tool_select") or "",
            ]
            new_sig = "||".join(signature_parts)
            if st.session_state.get("last_input_signature") != new_sig:
                _clear_generation_state()
                st.session_state["last_input_signature"] = new_sig
            with st.spinner("Generating..."):
                used_hybrid_flag = False
                actual = None

                # When the checkbox is enabled in any mode, force hybrid-style generation (generate both)
                if gen_both_and_choose and model_choice in ("Offline", "Gemini"):
                    offline_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                    gemini_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    offline_obj = save_prompt(tool, content_type, topic, style, offline_text, model_used="offline", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)
                    gemini_obj = save_prompt(tool, content_type, topic, style, gemini_text, model_used="gemini", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)

                    used_hybrid_flag = True
                    st.session_state["awaiting_hybrid_choice"] = True
                    st.session_state["hybrid_offline_text"] = offline_text
                    st.session_state["hybrid_gemini_text"] = gemini_text
                    st.session_state["hybrid_offline_id"] = offline_obj.id
                    st.session_state["hybrid_gemini_id"] = gemini_obj.id
                    st.session_state["show_output"] = False

                elif model_choice == "Hybrid":
                    # Always generate both variants for Hybrid
                    offline_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                    gemini_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    offline_obj = save_prompt(tool, content_type, topic, style, offline_text, model_used="offline", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)
                    gemini_obj = save_prompt(tool, content_type, topic, style, gemini_text, model_used="gemini", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)

                    used_hybrid_flag = True

                    if gen_both_and_choose:
                        # Ask the user to choose
                        st.session_state["awaiting_hybrid_choice"] = True
                        st.session_state["hybrid_offline_text"] = offline_text
                        st.session_state["hybrid_gemini_text"] = gemini_text
                        st.session_state["hybrid_offline_id"] = offline_obj.id
                        st.session_state["hybrid_gemini_id"] = gemini_obj.id
                        st.session_state["show_output"] = False
                    else:
                        # Auto-pick using ensemble scoring (Phase 5) → ranker fallback
                        auto_chosen = False

                        # --- Phase 5: Ensemble Prompt Synthesis ---
                        try:
                            from ensemble_synthesis import ensemble_select
                            import yaml as _yaml

                            try:
                                with open("config.yaml", "r", encoding="utf-8") as _cf:
                                    _ens_config = _yaml.safe_load(_cf)
                            except Exception:
                                _ens_config = {}

                            ens_slots = {
                                "tool": tool,
                                "content_type": content_type,
                                "topic": topic,
                                "style": style,
                                "platform": platform,
                                "color_palette": color_palette,
                                "mood": mood,
                            }
                            ens_source, ens_text, ens_log = ensemble_select(
                                offline_text, gemini_text,
                                ens_slots, _ens_config,
                            )
                            ai_choice = ens_source  # "offline" or "gemini"
                            chosen_id = offline_obj.id if ai_choice == "offline" else gemini_obj.id
                            uid = st.session_state["user"]["id"] if st.session_state.get("user") else None
                            record_choice(uid, offline_obj.id, gemini_obj.id, chosen_id, ai_choice)
                            with SessionLocal() as session:
                                saved = session.get(Prompt, chosen_id)
                            st.session_state["show_output"] = True
                            st.session_state["generated_text"] = saved.generated_text
                            st.session_state["generated_prompt_id"] = saved.id
                            st.session_state["ensemble_log"] = ens_log
                            auto_chosen = True
                        except Exception:
                            pass  # ensemble not available — try ranker

                        # --- Ranker fallback ---
                        if not auto_chosen:
                            try:
                                from ranker import load_ranker, predict_with_ranker
                                have_ranker = load_ranker(RANKER_PATH) is not None
                            except Exception:
                                have_ranker = False

                            if have_ranker:
                                topic_val = topic or ""
                                style_val = style or ""
                                txt = f"{topic_val} | {style_val} | OFFLINE: {offline_text} || GEMINI: {gemini_text}"
                                pred, _ = predict_with_ranker(txt, RANKER_PATH)
                                ai_choice = pred if pred in ("offline", "gemini") else random.choice(["offline", "gemini"])
                                chosen_id = offline_obj.id if ai_choice == "offline" else gemini_obj.id
                                uid = st.session_state["user"]["id"] if st.session_state.get("user") else None
                                record_choice(uid, offline_obj.id, gemini_obj.id, chosen_id, ai_choice)
                                with SessionLocal() as session:
                                    saved = session.get(Prompt, chosen_id)
                                st.session_state["show_output"] = True
                                st.session_state["generated_text"] = saved.generated_text
                                st.session_state["generated_prompt_id"] = saved.id
                                auto_chosen = True

                        # --- No auto-pick available: ask the user ---
                        if not auto_chosen:
                            st.warning("No trained ranker found. Please choose a variant or train the ranker in Admin → Train Ranker.")
                            st.session_state["awaiting_hybrid_choice"] = True
                            st.session_state["hybrid_offline_text"] = offline_text
                            st.session_state["hybrid_gemini_text"] = gemini_text
                            st.session_state["hybrid_offline_id"] = offline_obj.id
                            st.session_state["hybrid_gemini_id"] = gemini_obj.id
                            st.session_state["show_output"] = False
                else:
                    actual = model_choice.lower()

                if model_choice != "Hybrid" and not used_hybrid_flag:
                    if actual == "gemini":
                        prompt_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                        model_used = "gemini"
                    else:
                        prompt_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                        model_used = "offline"

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    saved = save_prompt(tool, content_type, topic, style, prompt_text, model_used=model_used, user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=used_hybrid_flag)
                    st.session_state["show_output"] = True
                    st.session_state["generated_text"] = saved.generated_text
                    st.session_state["generated_prompt_id"] = saved.id

            if st.session_state.get("awaiting_hybrid_choice"):
                st.info("Two variants generated. Please choose one or let AI choose.")

        # When awaiting choice, render both prompts and provide actions
        if st.session_state.get("awaiting_hybrid_choice"):
            st.subheader("Compare Variants")
            st.markdown("---")
            st.markdown("**Offline variant:**")
            st.code(st.session_state.get("hybrid_offline_text") or "", language="markdown")
            st.markdown("**Gemini variant:**")
            st.code(st.session_state.get("hybrid_gemini_text") or "", language="markdown")

            col_left, col_mid, col_ai = st.columns([1, 1, 1])

            def _choose(model: str):
                uid = st.session_state["user"]["id"] if st.session_state.get("user") else None
                offline_id = st.session_state.get("hybrid_offline_id")
                gemini_id = st.session_state.get("hybrid_gemini_id")
                chosen_id = offline_id if model == "offline" else gemini_id
                ok, res = record_choice(uid, offline_id, gemini_id, chosen_id, model)
                if not ok:
                    st.error("Could not record choice: " + (res or "unknown reason"))
                    return
                # Show chosen result
                with SessionLocal() as session:
                    saved = session.get(Prompt, chosen_id)
                _clear_generation_state()
                st.session_state["show_output"] = True
                st.session_state["generated_text"] = saved.generated_text
                st.session_state["generated_prompt_id"] = saved.id
                st.success("Choice recorded — thanks!")

            with col_left:
                if st.button("✅ Keep Offline", key="keep_offline_btn"):
                    _choose("offline")
                    safe_rerun()
            with col_mid:
                if st.button("✅ Keep Gemini", key="keep_gemini_btn"):
                    _choose("gemini")
                    safe_rerun()
            with col_ai:
                if st.button("🤖 Let AI Choose", key="ai_choose_btn"):
                    with st.spinner("Letting AI decide..."):
                        ai_choice = None
                        ens_log = None

                        # --- Try ensemble first ---
                        try:
                            from ensemble_synthesis import ensemble_select
                            import yaml as _yaml
                            try:
                                with open("config.yaml", "r", encoding="utf-8") as _cf:
                                    _ens_config = _yaml.safe_load(_cf)
                            except Exception:
                                _ens_config = {}

                            ens_slots = {
                                "tool": st.session_state.get("tool_select"),
                                "content_type": st.session_state.get("content_type_input"),
                                "topic": st.session_state.get("topic_input"),
                                "style": st.session_state.get("style_input"),
                                "platform": st.session_state.get("canva_platform"),
                                "color_palette": st.session_state.get("canva_colors"),
                                "mood": st.session_state.get("canva_mood"),
                            }
                            ens_source, ens_text, ens_log = ensemble_select(
                                st.session_state.get("hybrid_offline_text") or "",
                                st.session_state.get("hybrid_gemini_text") or "",
                                ens_slots, _ens_config,
                            )
                            ai_choice = ens_source
                        except Exception:
                            pass  # ensemble not available — try ranker

                        # --- Ranker fallback ---
                        if ai_choice is None:
                            try:
                                from ranker import load_ranker, predict_with_ranker
                                have_ranker = load_ranker(RANKER_PATH) is not None
                            except Exception:
                                have_ranker = False

                            if have_ranker:
                                topic_val = st.session_state.get("topic_input") or ""
                                style_val = st.session_state.get("style_input") or ""
                                txt = f"{topic_val} | {style_val} | OFFLINE: {st.session_state.get('hybrid_offline_text') or ''} || GEMINI: {st.session_state.get('hybrid_gemini_text') or ''}"
                                try:
                                    pred, _ = predict_with_ranker(txt, RANKER_PATH)
                                    ai_choice = str(pred) if pred in ("offline", "gemini") else None
                                except Exception:
                                    ai_choice = None

                    if ai_choice is None:
                        st.warning("No trained ranker available or prediction failed. Please choose manually or train the ranker in Admin → Train Ranker.")
                    else:
                        if ens_log:
                            st.session_state["ensemble_log"] = ens_log
                        _choose(ai_choice)
                        safe_rerun()

        if st.session_state.get("show_output") and st.session_state.get("generated_text"):
            st.markdown("---")
            st.subheader("Generated Prompt")
            st.success("✅ Prompt Generated & Saved:")
            st.code(st.session_state["generated_text"], language="markdown")

            # Show ensemble scoring details if available
            if st.session_state.get("ensemble_log"):
                with st.expander("📊 Ensemble Scoring Details", expanded=False):
                    for entry in st.session_state["ensemble_log"]:
                        cols = st.columns([1, 1, 1, 1, 1])
                        cols[0].metric("Source", entry.get("source", "—"))
                        cols[1].metric("Slot Score", f"{entry.get('slot_score', 0):.4f}")
                        cols[2].metric("Fluency Score", f"{entry.get('fluency_score', 0):.4f}")
                        cols[3].metric("Final Score", f"{entry.get('final_score', 0):.4f}")
                        cols[4].metric("Selected", "✅" if entry.get("selected") else "—")
                        st.markdown("---")

    # History tab
    with tab2:
        if not st.session_state.get("user"):
            st.info("Please login to view your prompt history.")
            return

        st.subheader("📂 Search Your Prompt History")
        search_tool = st.selectbox("Filter by tool:", ["All", "Gamma", "Canva"], key="search_tool_select")
        search_topic = st.text_input("Search by topic:", key="search_topic_input")
        col_btn1, col_btn2, _ = st.columns([0.8, 0.8, 8])
        search_clicked = col_btn1.button("🔎 Search", key="btn_search")
        view_all_clicked = col_btn2.button("📂 View All", key="btn_view_all")

        if search_clicked:
            st.session_state["last_action"] = "search"
            st.session_state["last_search_tool"] = search_tool
            st.session_state["last_search_topic"] = search_topic
        elif view_all_clicked:
            st.session_state["last_action"] = "view_all"

        results = []
        with SessionLocal() as session:
            current_user_id = st.session_state["user"]["id"]
            if st.session_state["last_action"] == "search":
                stmt = select(Prompt).where(Prompt.user_id == current_user_id)
                if st.session_state["last_search_tool"] != "All":
                    stmt = stmt.where(Prompt.tool == st.session_state["last_search_tool"])
                if st.session_state["last_search_topic"]:
                    stmt = stmt.where(Prompt.topic.contains(st.session_state["last_search_topic"]))
                results = session.execute(stmt).scalars().all()
            elif st.session_state["last_action"] == "view_all":
                stmt = select(Prompt).where(Prompt.user_id == current_user_id)
                results = session.execute(stmt).scalars().all()

        if results:
            with st.expander("🔎 Show Results", expanded=True):
                for r in results:
                    st.markdown(
                        f"**Tool:** {r.tool} | **Topic:** {r.topic} | **Style:** {r.style} | **Model:** {r.model_used}"
                    )
                    if r.platform_name or r.color_palette or r.mood:
                        meta = []
                        if r.platform_name:
                            meta.append(f"Platform: {r.platform_name}")
                        if r.color_palette:
                            meta.append(f"Colors: {r.color_palette}")
                        if r.mood:
                            meta.append(f"Mood: {r.mood}")
                        st.caption(" • ".join(meta))

                    st.code(r.generated_text, language="markdown")
                    colA, colB, colC = st.columns([1.1, 1.8, 8])
                    with colA:
                        st.button(
                            label="🗑️ Delete",
                            key=f"delete_btn_{r.id}",
                            on_click=handle_delete,
                            args=(r.id,)
                        )
                    with colB:
                        st.download_button(
                            label="📥 Download as .txt",
                            data=r.generated_text,
                            file_name=f"prompt_{r.id}.txt",
                            mime="text/plain",
                            key=f"download_{r.id}"
                        )
                    st.markdown("---")
        else:
            if st.session_state["last_action"] in ("search", "view_all"):
                st.info("No prompts found matching your filters or there were no prompts generated.")


# App entry
def main():
    st.set_page_config(page_title="Prompt Gen", page_icon="✨", layout="wide")
    app_mode = st.sidebar.radio("Navigation", ["Prompt Generator", "Admin Panel"], index=0)

    if app_mode == "Prompt Generator":
        run_prompt_generator()
    else:
        if st.session_state.get("admin_logged_in", False):
            if st.sidebar.button("Logout Admin"):
                st.session_state["admin_logged_in"] = False
                safe_rerun()
            admin_panel()
        else:
            logged_in_now = admin_login_ui()
            if logged_in_now:
                admin_panel()

if __name__ == "__main__":
    main()
