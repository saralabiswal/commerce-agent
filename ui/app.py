"""
CommerceAgent — Streamlit Dashboard
Provides a full UI for optimizing product listings, viewing results,
run history, metrics, and submitting human feedback.
"""
import asyncio
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CommerceAgent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run async code from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def score_color(score: float) -> str:
    if score >= 80:
        return "#22c55e"
    if score >= 70:
        return "#f59e0b"
    return "#ef4444"


def grade_emoji(grade: str) -> str:
    return {"A": "🏆", "B": "✅", "C": "⚠️", "D": "🔴", "F": "❌"}.get(grade, "")


def severity_badge(severity: str) -> str:
    colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#f59e0b", "low": "#6b7280"}
    color = colors.get(severity, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{severity.upper()}</span>'


def render_page_header(title: str, description: str) -> None:
    """Render a compact, enterprise-style page header with business context."""
    st.markdown(f"### {title}")
    st.caption(description)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 CommerceAgent")
    st.caption("AI-powered listing optimization")
    st.caption("Author: Sarala Biswal")
    st.divider()

    page = st.radio(
        "Navigation",
        ["About", "Optimize", "Audit Only", "Run History", "Metrics", "Settings", "Architecture"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Provider Config**")
    try:
        from config import settings
        st.caption(f"LLM: `{settings.llm_provider}`")
        if settings.llm_provider == "anthropic":
            st.caption(f"Model: `{settings.anthropic_model.split('-')[1]}`")
        st.caption(f"Quality gate: `{settings.quality_gate_threshold}`")
        st.caption(f"Max retries: `{settings.quality_gate_max_retries}`")
    except Exception:
        st.caption("Config not loaded")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OPTIMIZE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Optimize":
    render_page_header(
        "Full Optimization Pipeline",
        "Audit the current listing, compare market patterns, generate retailer-ready content, "
        "and return a scored recommendation for the selected product and retailer.",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            from mcp_servers.catalog_mcp_server import list_products

            products = list_products()
        except Exception:
            products = []

        if products:
            labels_by_sku = {
                product["sku"]: product.get("product_name") or product["sku"]
                for product in products
            }
            selected_sku = st.selectbox(
                "Product",
                options=list(labels_by_sku.keys()),
                format_func=lambda product_sku: labels_by_sku[product_sku],
            )
            sku = selected_sku
            st.caption(f"SKU / ASIN: `{sku}`")
        else:
            sku = st.text_input(
                "Product SKU / ASIN",
                value="ANKER-Q30-BLK",
                placeholder="e.g. ANKER-Q30-BLK or B08N5WRWNW",
            )
    with col2:
        retailer = st.selectbox("Retailer", ["amazon", "walmart"])

    st.info(
        "Use this workflow when a listing needs a full business review. CommerceAgent audits "
        "the current content, studies competitor signals, retrieves retailer guidance, then "
        "generates and scores optimized copy before showing results."
    )
    run_btn = st.button("⚡ Run Optimization", type="primary", use_container_width=True)

    if run_btn and sku:
        progress_area = st.empty()
        progress_events = []
        progress_steps = {
            "Pipeline": 5,
            "ContentAuditAgent": 25,
            "CompetitorAnalysisAgent": 45,
            "ContentGenerationAgent": 70,
            "Quality Gate": 90,
        }

        def update_progress(stage: str, message: str):
            progress_events.append((stage, message))
            with progress_area.container():
                st.progress(progress_steps.get(stage, 10), text=f"{stage}: {message}")
                for event_stage, event_message in progress_events[-4:]:
                    st.caption(f"**{event_stage}** — {event_message}")

        update_progress("Pipeline", "Starting optimization...")

        try:
            from inspect import signature

            from agents.orchestrator import CommerceAgentOrchestrator
            from llm.factory import get_llm_provider

            provider = get_llm_provider()
            orchestrator = CommerceAgentOrchestrator(provider)
            run_kwargs = {"sku": sku, "retailer": retailer}
            if "progress_callback" in signature(orchestrator.run).parameters:
                run_kwargs["progress_callback"] = update_progress
            result = run_async(orchestrator.run(**run_kwargs))
            st.session_state["last_result"] = result
            st.session_state["last_sku"] = sku
            st.session_state["last_retailer"] = retailer
            st.session_state["last_progress_events"] = progress_events
            progress_area.empty()
            st.success(f"✅ Optimization complete — Run ID: `{result.run_id}`")
            with st.expander("Optimization progress", expanded=False):
                for event_stage, event_message in progress_events:
                    st.caption(f"**{event_stage}** — {event_message}")
        except Exception as e:
            progress_area.empty()
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            if progress_events:
                with st.expander("Optimization progress", expanded=False):
                    for event_stage, event_message in progress_events:
                        st.caption(f"**{event_stage}** — {event_message}")

    # ── Display results ───────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        audit = result.audit
        competitors = result.competitors
        content = result.content

        st.divider()

        # Top-level score cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Score Before", f"{audit.current_score:.0f}/100")
        with m2:
            delta = content.quality_score - audit.current_score
            st.metric("Score After", f"{content.quality_score:.0f}/100", delta=f"{delta:+.0f}")
        with m3:
            st.metric("Latency", f"{result.total_latency_ms/1000:.1f}s")
        with m4:
            st.metric("Retries", content.retry_count)
        with m5:
            st.metric("Est. Cost", f"${result.estimated_cost_usd:.4f}")

        # Content warnings
        if content.warnings:
            with st.expander("⚠️ Warnings", expanded=True):
                for w in content.warnings:
                    st.warning(w)

        tab1, tab2, tab3, tab4 = st.tabs(["📝 Generated Content", "🔍 Audit Report", "📊 Competitors", "📈 Score Breakdown"])

        # ── Tab 1: Generated Content ──────────────────────────────────────────
        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Optimized Listing")
                st.markdown(f"**Title** `{len(content.title)} chars`")
                st.info(content.title)

                st.markdown("**Bullet Points**")
                for i, bullet in enumerate(content.bullet_points, 1):
                    st.markdown(f"`{i}.` {bullet}")
                    st.caption(f"{len(bullet)} chars")

                st.markdown("**Description**")
                st.text_area(
                    "Generated description",
                    value=content.description,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )

                st.markdown("**Backend Keywords**")
                st.code(content.backend_keywords)

            with c2:
                st.subheader("Quality & Compliance")
                grade = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "🔴", "F": "❌"}.get(
                    ("A" if content.quality_score >= 90 else
                     "B" if content.quality_score >= 80 else
                     "C" if content.quality_score >= 70 else
                     "D" if content.quality_score >= 60 else "F"), "")

                st.markdown(
                    f"<div style='text-align:center;padding:16px;background:#1e293b;border-radius:8px'>"
                    f"<div style='font-size:48px'>{grade}</div>"
                    f"<div style='font-size:32px;font-weight:bold;color:{score_color(content.quality_score)}'>"
                    f"{content.quality_score:.1f}</div>"
                    f"<div style='color:#94a3b8'>Quality Score</div></div>",
                    unsafe_allow_html=True
                )

                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    icon = "✅" if content.compliance_check.get("compliant") else "❌"
                    st.markdown(f"{icon} **Compliance**")
                with col_b:
                    icon = "✅" if content.brand_safety.get("passed") else "❌"
                    st.markdown(f"{icon} **Brand Safety**")

                st.divider()
                if content.reasoning:
                    st.markdown("**AI Reasoning**")
                    st.caption(content.reasoning)

                # Feedback buttons
                st.divider()
                st.markdown("**Was this helpful?**")
                fb1, fb2 = st.columns(2)
                with fb1:
                    if st.button("👍 Good output", use_container_width=True):
                        try:
                            from evaluation.human_feedback import HumanFeedbackStore
                            store = HumanFeedbackStore()
                            run_async(store.record_feedback(
                                run_id=result.run_id, sku=sku,
                                rating=1, field="overall"
                            ))
                            st.success("Thanks!")
                        except Exception as e:
                            st.error(str(e))
                with fb2:
                    if st.button("👎 Needs work", use_container_width=True):
                        try:
                            from evaluation.human_feedback import HumanFeedbackStore
                            store = HumanFeedbackStore()
                            run_async(store.record_feedback(
                                run_id=result.run_id, sku=sku,
                                rating=-1, field="overall"
                            ))
                            st.info("Feedback recorded.")
                        except Exception as e:
                            st.error(str(e))

        # ── Tab 2: Audit Report ───────────────────────────────────────────────
        with tab2:
            st.subheader(f"Audit: {audit.sku} on {audit.retailer}")
            st.metric("Current Score", f"{audit.current_score:.0f}/100")

            if audit.priority_improvements:
                st.markdown("**Priority Improvements**")
                for p in audit.priority_improvements:
                    if "[COMPLIANCE]" in p or "[CRITICAL]" in p:
                        st.error(p)
                    elif "[HIGH]" in p:
                        st.warning(p)
                    else:
                        st.info(p)

            if audit.gap_analysis:
                st.divider()
                st.markdown("**Gap Analysis**")
                for gap in audit.gap_analysis:
                    with st.expander(f"{gap.field.upper()}: {gap.issue[:60]}..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Severity:** {gap.severity}")
                            st.markdown(f"**Current:** {gap.current_value[:100] if gap.current_value else 'N/A'}")
                        with col2:
                            st.markdown(f"**Fix:** {gap.recommendation}")

            # Character counts table
            if audit.character_counts:
                st.divider()
                st.markdown("**Character Utilization**")
                for field_name, counts in audit.character_counts.items():
                    current = counts.get("current", 0)
                    allowed = counts.get("allowed", 0)
                    if allowed:
                        pct = current / allowed
                        color = "#22c55e" if pct <= 1.0 else "#ef4444"
                        st.markdown(
                            f"**{field_name}**: `{current}/{allowed}` chars "
                            f"<span style='color:{color}'>{pct*100:.0f}%</span>",
                            unsafe_allow_html=True
                        )

        # ── Tab 3: Competitors ────────────────────────────────────────────────
        with tab3:
            st.subheader("Competitor Intelligence")
            st.caption(f"Analyzed {competitors.competitor_count} competitors in {competitors.category}")

            if competitors.top_keywords:
                st.markdown("**Top Keywords by Search Volume**")
                kw_data = []
                for kw in competitors.top_keywords:
                    kw_data.append({
                        "Keyword": kw.term,
                        "Monthly Searches": f"{kw.monthly_volume:,}",
                        "Competition": kw.competition,
                        "In Our Listing": "✅" if kw.present_in_listing else "❌",
                    })
                st.table(kw_data)

            col_left, col_right = st.columns(2)
            with col_left:
                if competitors.winning_patterns:
                    st.markdown("**Winning Patterns (What Top Performers Do)**")
                    for i, pattern in enumerate(competitors.winning_patterns, 1):
                        st.markdown(f"{i}. {pattern}")

            with col_right:
                if competitors.content_gaps:
                    st.markdown("**Content Gaps (What We're Missing)**")
                    for i, gap in enumerate(competitors.content_gaps, 1):
                        st.markdown(f"{i}. {gap}")

            if competitors.benchmark_scores:
                st.divider()
                st.markdown("**Competitor Quality Scores**")
                for brand, score in sorted(competitors.benchmark_scores.items(), key=lambda x: -x[1]):
                    bar_width = int(score)
                    color = score_color(score)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                        f"<span style='width:140px;font-size:13px'>{brand}</span>"
                        f"<div style='flex:1;background:#1e293b;border-radius:4px;height:18px'>"
                        f"<div style='width:{bar_width}%;background:{color};height:100%;border-radius:4px'></div></div>"
                        f"<span style='font-weight:bold;color:{color}'>{score:.0f}</span></div>",
                        unsafe_allow_html=True
                    )

        # ── Tab 4: Score Breakdown ────────────────────────────────────────────
        with tab4:
            st.subheader("Score Breakdown")
            if content.score_breakdown:
                for dim, score in content.score_breakdown.items():
                    color = score_color(score)
                    bar = int(score)
                    label = dim.replace("_", " ").title()
                    st.markdown(
                        f"<div style='margin:8px 0'>"
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<span>{label}</span><span style='color:{color};font-weight:bold'>{score:.0f}/100</span></div>"
                        f"<div style='background:#1e293b;border-radius:4px;height:12px;margin-top:4px'>"
                        f"<div style='width:{bar}%;background:{color};height:100%;border-radius:4px'></div></div></div>",
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT ONLY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audit Only":
    render_page_header(
        "Content Audit",
        "Assess listing quality before generation. Use this to understand compliance gaps, "
        "content weaknesses, and the highest-priority fixes for a product page.",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            from mcp_servers.catalog_mcp_server import list_products

            products = list_products()
        except Exception:
            products = []

        if products:
            labels_by_sku = {
                product["sku"]: product.get("product_name") or product["sku"]
                for product in products
            }
            selected_sku = st.selectbox(
                "Product",
                options=list(labels_by_sku.keys()),
                format_func=lambda product_sku: labels_by_sku[product_sku],
                key="audit_product",
            )
            sku = selected_sku
            st.caption(f"SKU / ASIN: `{sku}`")
        else:
            sku = st.text_input("SKU / ASIN", value="ANKER-Q30-BLK")
    with col2:
        retailer = st.selectbox("Retailer", ["amazon", "walmart"])

    st.info(
        "This action does not create new copy. It gives business and content teams a fast read "
        "on what is working, what is missing, and what needs attention before optimization."
    )
    if st.button("Run Audit", type="primary"):
        with st.spinner("Auditing..."):
            try:
                from agents.content_audit_agent import ContentAuditAgent
                from llm.factory import get_llm_provider
                provider = get_llm_provider()
                agent = ContentAuditAgent(provider)
                audit = run_async(agent.run(sku=sku, retailer=retailer))

                st.metric("Quality Score", f"{audit.current_score:.1f}/100")

                st.divider()
                st.subheader("Priority Improvements")
                for p in audit.priority_improvements:
                    st.markdown(f"- {p}")

                st.divider()
                st.subheader("Gap Analysis")
                for gap in audit.gap_analysis:
                    st.markdown(f"**{gap.field.upper()}** — {gap.issue}")
                    st.caption(f"→ {gap.recommendation}")

            except Exception as e:
                st.error(f"Audit failed: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Run History":
    render_page_header(
        "Run History",
        "Review previous optimization runs, score movement, latency, quality gate status, "
        "and the model/provider used for each listing workflow.",
    )
    st.info(
        "Use this page for operational review: compare outcomes across products, confirm whether "
        "quality gates passed, and identify runs that may need human follow-up."
    )

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        from observability.tracing import get_tracer
        runs = run_async(get_tracer().get_recent_runs(limit=50))

        if not runs:
            st.info("No runs yet. Run an optimization on the Optimize page!")
        else:
            # Summary row
            avg_score = sum(r["score_after"] for r in runs) / len(runs)
            avg_delta = sum(r["improvement_delta"] for r in runs) / len(runs)
            total_cost = sum(r["estimated_cost_usd"] for r in runs)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Runs", len(runs))
            c2.metric("Avg Score", f"{avg_score:.1f}")
            c3.metric("Avg Improvement", f"{avg_delta:+.1f}")
            c4.metric("Total Cost", f"${total_cost:.4f}")

            st.divider()

            # Run table
            for run in runs:
                with st.expander(
                    f"[{run['run_id']}] {run['sku']} on {run['retailer']} — "
                    f"Score: {run['score_before']:.0f} → {run['score_after']:.0f} "
                    f"({'✅' if run['quality_passed'] else '⚠️'})",
                    expanded=False,
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Score Before", f"{run['score_before']:.0f}")
                    col2.metric("Score After", f"{run['score_after']:.0f}", delta=f"{run['improvement_delta']:+.0f}")
                    col3.metric("Latency", f"{run['latency_ms']/1000:.1f}s")
                    col4.metric("Cost", f"${run['estimated_cost_usd']:.4f}")
                    st.caption(
                        f"Provider: `{run['provider']}` | Model: `{run['model']}` | "
                        f"Timestamp: `{run['timestamp'][:19]}`"
                    )

    except Exception as e:
        st.error(f"Could not load run history: {e}")
        st.info("Initialize the database with `python -m scripts.init_db`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Metrics":
    render_page_header(
        "Platform Metrics",
        "Track pipeline health across runs, including average quality, improvement, latency, "
        "cost, pass rate, and human feedback trends.",
    )
    st.info(
        "This page is designed for business and platform owners who need to know whether the "
        "optimization process is improving listings reliably and efficiently over time."
    )

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        from observability.tracing import get_tracer
        metrics = run_async(get_tracer().get_aggregate_metrics())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Runs", metrics["total_runs"])
        c2.metric("Avg Quality Score", f"{metrics['avg_quality_score']:.1f}/100")
        c3.metric("Avg Improvement", f"{metrics['avg_improvement']:+.1f} pts")

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Latency", f"{metrics['avg_latency_ms']/1000:.1f}s")
        c5.metric("Total Cost", f"${metrics['total_cost_usd']:.4f}")
        c6.metric("Quality Gate Pass Rate", f"{metrics['quality_gate_pass_rate']:.0f}%")

        # Feedback summary
        st.divider()
        st.subheader("Human Feedback")
        try:
            from evaluation.human_feedback import HumanFeedbackStore
            store = HumanFeedbackStore()
            feedback = run_async(store.get_summary())
            f1, f2, f3 = st.columns(3)
            f1.metric("Total Feedback", feedback["total_feedback"])
            f2.metric("Thumbs Up", feedback["thumbs_up"])
            f3.metric("Approval Rate", f"{feedback['approval_rate']:.0f}%")
        except Exception:
            st.info("No feedback recorded yet.")

        # Score distribution from run history
        st.divider()
        st.subheader("Score Distribution")
        st.caption(
            "This compares listing quality before and after optimization across recent runs. "
            "A healthy pipeline should move more listings toward the higher score ranges after generation."
        )
        try:
            runs = run_async(get_tracer().get_recent_runs(limit=100))
            if runs:
                import plotly.graph_objects as go
                scores_before = [r["score_before"] for r in runs]
                scores_after = [r["score_after"] for r in runs]
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=scores_before, name="Before", opacity=0.7, nbinsx=10))
                fig.add_trace(go.Histogram(x=scores_after, name="After", opacity=0.7, nbinsx=10))
                fig.update_layout(
                    barmode="overlay",
                    xaxis_title="Quality Score",
                    yaxis_title="Count",
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Improvement over time
                if len(runs) >= 3:
                    st.caption(
                        "The trend below shows score lift by run date. Positive values mean the "
                        "optimized content scored better than the original listing."
                    )
                    fig2 = go.Figure()
                    timestamps = [r["timestamp"][:10] for r in reversed(runs[:20])]
                    improvements = [r["improvement_delta"] for r in reversed(runs[:20])]
                    fig2.add_trace(go.Scatter(x=timestamps, y=improvements, mode="lines+markers", name="Improvement Delta"))
                    fig2.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Score Improvement",
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("Run some optimizations to see score distribution.")

    except Exception as e:
        st.error(f"Could not load metrics: {e}")
        st.info("Initialize the database with `python -m scripts.init_db`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    render_page_header(
        "Settings",
        "Review active runtime configuration and validate provider/RAG readiness before running "
        "business workflows.",
    )

    try:
        from config import settings

        st.subheader("Current Configuration")
        config_data = {
            "LLM Provider": settings.llm_provider,
            "Model (Anthropic)": settings.anthropic_model,
            "Model (Ollama)": settings.ollama_model,
            "Quality Gate Threshold": settings.quality_gate_threshold,
            "Max Retries": settings.quality_gate_max_retries,
            "RAG Mode": settings.rag_mode,
            "RAG Top-K": settings.rag_top_k,
            "Embedding Model": settings.embedding_model,
            "LangSmith Tracing": settings.langchain_tracing_v2,
            "Guardrails Enabled": settings.guardrails_enabled,
        }
        for k, v in config_data.items():
            col1, col2 = st.columns([1, 2])
            col1.markdown(f"**{k}**")
            col2.code(str(v))

        st.divider()
        st.subheader("Provider Health Check")
        st.info(
            "Checks whether the active LLM provider can respond successfully. Run this when "
            "switching providers, changing model settings, or troubleshooting slow/failed runs."
        )
        if st.button("Test Provider Connection"):
            with st.spinner("Testing..."):
                try:
                    from llm.factory import get_llm_provider
                    provider = get_llm_provider()
                    result = run_async(provider.health_check())
                    if result["status"] == "ok":
                        st.success(
                            f"✅ {result['provider']} / {result['model']} — "
                            f"latency: {result.get('latency_ms', 0):.0f}ms"
                        )
                    else:
                        st.error(f"❌ Provider error: {result.get('error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        st.divider()
        st.subheader("RAG Status")
        st.info(
            "Confirms that retailer guidance is indexed and retrievable. Healthy RAG helps the "
            "generation step stay grounded in marketplace rules instead of relying on generic copy."
        )
        if st.button("Check RAG"):
            with st.spinner("Checking ChromaDB..."):
                try:
                    from rag.retrieval import get_retriever
                    retriever = get_retriever()
                    count = retriever.document_count
                    st.success(f"✅ ChromaDB ready — {count} chunks indexed")

                    sample = retriever.retrieve("amazon title requirements electronics", top_k=2)
                    if sample:
                        st.markdown("**Sample retrieval (title requirements):**")
                        for chunk in sample:
                            with st.expander(f"Source: {chunk['source']} (score: {chunk['relevance_score']:.3f})"):
                                st.text(chunk["text"][:300] + "...")
                except Exception as e:
                    st.error(f"RAG check failed: {e}")

        st.divider()
        st.subheader("Re-ingest RAG Documents")
        st.warning(
            "Use re-ingestion after changing files in `rag/documents/`. This refreshes the local "
            "retrieval index used by the content generation agent."
        )
        if st.button("Re-ingest Documents", help="Re-process all documents in rag/documents/"):
            with st.spinner("Ingesting..."):
                try:
                    from rag.ingestion import ingest
                    from rag.retrieval import reset_retriever

                    count = ingest(verbose=False)
                    reset_retriever()
                    st.success(f"✅ Ingested {count} chunks")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    except Exception as e:
        st.error(f"Settings error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(
        """
        <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:28px 30px;border-radius:8px;margin-bottom:22px;">
          <div style="color:#f59e0b;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">
            CommerceAgent
          </div>
          <div style="color:#111827;font-size:30px;font-weight:750;line-height:1.18;margin-bottom:10px;">
            Your product listings are losing sales. Every day.
          </div>
          <div style="color:#475569;font-size:15px;line-height:1.55;max-width:880px;">
            CommerceAgent finds exactly what's wrong, benchmarks you against the competition,
            and rewrites your content — in seconds.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("The Problem")
    st.markdown(
        "If you sell on Amazon or Walmart, your product listing is your storefront. "
        "A weak title, vague bullet points, or a missing keyword can drop you 20 "
        "positions in search — costing you thousands in lost sales. Most brands "
        "don't find out until it's already happened."
    )
    problem_cards = [
        (
            "📉",
            "Listings Degrade Over Time",
            "Titles get truncated. Keywords go stale. Competitor content improves while yours "
            "stays the same. A listing that ranked #3 last year might rank #23 today — with "
            "nothing visibly wrong.",
        ),
        (
            "⏱️",
            "Manual Auditing Doesn't Scale",
            "Reviewing 500 SKUs manually takes weeks, costs thousands, and produces inconsistent "
            "results. By the time your team finishes, the market has already moved. You need "
            "continuous intelligence, not quarterly reports.",
        ),
        (
            "🤖",
            "Generic AI Makes Things Worse",
            "Off-the-shelf AI tools write fluent content that violates retailer rules — wrong "
            "character counts, prohibited words, invented specs. A listing that gets suppressed "
            "by Amazon earns zero revenue, however well-written.",
        ),
    ]
    for column, (icon, title, body) in zip(st.columns(3), problem_cards):
        with column:
            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #ef4444;padding:18px;border-radius:8px;min-height:225px;">
                  <div style="font-size:22px;margin-bottom:8px;">{icon}</div>
                  <div style="color:#111827;font-size:16px;font-weight:700;margin-bottom:8px;">{title}</div>
                  <div style="color:#475569;font-size:13.5px;line-height:1.55;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.header("How CommerceAgent Works")
    st.markdown(
        "CommerceAgent runs three specialized AI agents in sequence — each one doing "
        "a job that would take a human analyst hours to complete."
    )
    solution_steps = [
        (
            "STEP 1 — DIAGNOSE",
            "🔍",
            "Audit Your Current Listing",
            "The system reads your live listing and scores it across five dimensions: title "
            "compliance, keyword coverage, bullet quality, readability, and specificity. It "
            "identifies exactly what's costing you rank — not a vague score, but a prioritized "
            "list of fixes with character counts, missing keywords, and compliance violations "
            "called out by name.",
        ),
        (
            "STEP 2 — BENCHMARK",
            "📊",
            "Analyze the Competition",
            "It pulls the top-performing competitor listings in your category and reverse-engineers "
            "what they're doing right — which keywords they rank for, how they structure their "
            "titles, what specifications they always include that you don't. You see your content "
            "gap clearly, measured in monthly search volume.",
        ),
        (
            "STEP 3 — REWRITE",
            "✍️",
            "Generate Optimized Content",
            "Armed with your audit and the competitive intelligence, the system generates a complete "
            "rewrite: title, five bullet points, description, and backend keywords — all grounded "
            "in retailer-specific rules retrieved in real time. The output is scored before it "
            "reaches you. If it doesn't meet the quality threshold, it rewrites automatically and "
            "tries again.",
        ),
    ]
    for label, icon, title, body in solution_steps:
        st.markdown(
            f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #22c55e;padding:18px;margin-bottom:12px;border-radius:8px;">
              <div style="color:#15803d;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">
                {label}
              </div>
              <div style="color:#111827;font-size:16px;font-weight:700;margin-bottom:8px;">{icon} {title}</div>
              <div style="color:#475569;font-size:13.5px;line-height:1.55;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.header("The Impact")
    st.markdown("This is what changes when your content is right.")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Content Quality", "58 → 84", delta="↑ 26 points", delta_color="normal")
    metric_2.metric("Time to Optimize", "< 60 sec", delta="vs. 4–6 hrs manual", delta_color="normal")
    metric_3.metric("Keywords Covered", "2 of 5 → 5 of 5", delta="Top search terms", delta_color="normal")
    metric_4.metric("Compliance", "Verified", delta="Before publish", delta_color="normal")
    st.success(
        "💡 The Anker Life Q30 example starts with a real-world weak listing — "
        "vague language, missing headphone specs, low keyword coverage — and ends with "
        "retailer-compliant, competitive content in under a minute."
    )

    st.divider()
    st.header("Who This Is For")
    persona_cards = [
        (
            "🏪",
            "Independent Sellers",
            "Amazon & Walmart stores, 50–500 SKUs",
            "You don't have a content team. You wrote your listings once and haven't touched "
            "them since. CommerceAgent tells you exactly which SKUs are underperforming and "
            "rewrites them in seconds — no agency fees, no guesswork.",
        ),
        (
            "🏭",
            "CPG Brands",
            "Multi-channel, 500+ SKUs",
            "Your content degrades across dozens of retailers while your team focuses on new "
            "launches. CommerceAgent runs continuous audits, flags compliance issues before "
            "they cause suppression, and keeps your content competitive at scale.",
        ),
        (
            "🏢",
            "E-Commerce Agencies",
            "Managing content for multiple clients",
            "Client audits that took days now take minutes. You can show clients exactly what's "
            "wrong, exactly how you'll fix it, and exactly how much the content improved — with "
            "scores and before/after comparisons built in.",
        ),
    ]
    for column, (icon, title, subtitle, body) in zip(st.columns(3), persona_cards):
        with column:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #e2e8f0;padding:18px;border-radius:8px;min-height:245px;">
                  <div style="font-size:22px;margin-bottom:8px;">{icon}</div>
                  <div style="color:#111827;font-size:16px;font-weight:700;margin-bottom:4px;">{title}</div>
                  <div style="color:#64748b;font-size:12.5px;font-weight:600;margin-bottom:10px;">{subtitle}</div>
                  <div style="color:#475569;font-size:13.5px;line-height:1.55;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.header("See It In Action")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#eff6ff,#f8fafc);border:1px solid #bfdbfe;padding:26px;border-radius:8px;text-align:center;margin-bottom:18px;">
          <div style="color:#111827;font-size:20px;font-weight:750;margin-bottom:8px;">Try it now with a real example</div>
          <div style="color:#475569;font-size:14px;line-height:1.6;max-width:760px;margin:0 auto;">
            Click Optimize in the sidebar. Select Soundcore by Anker Life Q30 Hybrid Active
            Noise Cancelling Headphones, Retailer: Amazon.
            Watch the three agents run in real time and see a weak listing become a competitive one.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    demo_left, demo_right = st.columns(2)
    with demo_left:
        st.info(
            "🔍 The audit will show you exactly what's wrong with the current listing "
            "and why it's underperforming."
        )
    with demo_right:
        st.success(
            "✅ The generated content will fix every issue — keyword gaps, vague language, "
            "spec omissions — and score it before delivery."
        )

    st.divider()
    st.markdown(
        """
        <div style="color:#64748b;font-size:13px;line-height:1.5;text-align:center;padding:8px 0 18px;">
          CommerceAgent is an open-source portfolio project by Sarala Biswal.
          Built to demonstrate production-grade agentic AI architecture applied
          to a real e-commerce problem. Not a SaaS product. Not a demo. A platform.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Architecture":
    from config import settings

    render_page_header(
        "Architecture",
        "Understand the agent workflow, tool boundaries, retrieval layer, quality gate, and "
        "observability design behind each optimization run.",
    )
    st.info(
        "CommerceAgent turns product-listing optimization into an auditable agent workflow. "
        "It is built for e-commerce sellers, CPG brands, and teams managing listings at scale."
    )

    st.header("Overview")
    st.markdown(
        "When you run an optimization, CommerceAgent audits the current listing, studies "
        "competitors, retrieves retailer rules, generates new content, and checks whether "
        "the result is good enough to ship. The system is model-agnostic, local-first for "
        "development, and structured so each agent owns one job."
    )

    st.header("How The Pipeline Works")
    steps = [
        (
            "1",
            "ContentAuditAgent",
            "Fetches the listing from RetailerMCP, scores it against retailer rules, "
            "and asks the LLM for a focused gap analysis.",
        ),
        (
            "2",
            "CompetitorAnalysisAgent",
            "Pulls top competitor listings from CatalogMCP, extracts keywords, enriches "
            "them with search volume, and uses the LLM to identify winning patterns.",
        ),
        (
            "3",
            "ContentGenerationAgent",
            "Fetches product specs from CatalogMCP, retrieves retailer guidance through "
            "RAG, builds a grounded prompt, generates content with the LLM, and scores it.",
        ),
        (
            "4",
            "Quality Gate",
            "If the score is below 70, LangGraph loops back to generation with specific "
            "repair instructions. The loop stops after 3 retries and returns the best output.",
        ),
    ]
    for number, name, detail in steps:
        with st.container():
            col_a, col_b = st.columns([1, 8])
            col_a.success(number)
            col_b.markdown(f"**{name}**")
            col_b.caption(detail)

    st.header("Architecture Diagram")
    st.graphviz_chart(
        """
        digraph CommerceAgent {
          graph [rankdir=LR, bgcolor="transparent", pad="0.3", nodesep="0.55", ranksep="0.75"];
          node [fontname="Helvetica", fontsize=11, style="rounded,filled", color="#94a3b8", fillcolor="#eef2ff"];
          edge [fontname="Helvetica", fontsize=10, color="#64748b"];

          User [shape=oval, fillcolor="#dcfce7"];
          FastAPI [shape=box, label="FastAPI"];
          Orchestrator [shape=box, label="LangGraph\\nOrchestrator", fillcolor="#e0f2fe"];
          Audit [shape=box, label="ContentAuditAgent"];
          Competitors [shape=box, label="CompetitorAnalysisAgent"];
          Generation [shape=box, label="ContentGenerationAgent"];
          QualityGate [shape=diamond, label="Quality\\nGate", fillcolor="#fef3c7"];
          End [shape=oval, label="END", fillcolor="#dcfce7"];

          RetailerMCP [shape=box, label="RetailerMCP", fillcolor="#f1f5f9"];
          CatalogMCP [shape=box, label="CatalogMCP", fillcolor="#f1f5f9"];
          ScoringMCP [shape=box, label="ScoringMCP", fillcolor="#f1f5f9"];
          RAG [shape=cylinder, label="RAG Pipeline\\nChromaDB", fillcolor="#fae8ff"];
          LLM [shape=box, label="LLM Provider\\nOllama / Anthropic / HF", fillcolor="#ede9fe"];
          Observability [shape=cylinder, label="Observability\\nLangSmith + SQLite", fillcolor="#fee2e2"];

          User -> FastAPI [label="1. submit"];
          FastAPI -> Orchestrator [label="2. start workflow"];
          Orchestrator -> Audit [label="3. audit"];
          Audit -> Competitors [label="4. analyze market"];
          Competitors -> Generation [label="5. generate"];
          Generation -> QualityGate [label="6. score"];
          QualityGate -> End [label="7. pass"];
          QualityGate -> Generation [label="8. fail / retry", color="#dc2626"];

          Audit -> RetailerMCP;
          Audit -> ScoringMCP;
          Competitors -> CatalogMCP;
          Competitors -> RetailerMCP;
          Generation -> CatalogMCP;
          Generation -> ScoringMCP;
          Generation -> RAG;

          Audit -> LLM [style=dashed];
          Competitors -> LLM [style=dashed];
          Generation -> LLM [style=dashed];

          Orchestrator -> Observability;
          Audit -> Observability [style=dotted];
          Competitors -> Observability [style=dotted];
          Generation -> Observability [style=dotted];
        }
        """
    )
    st.warning(
        "The quality gate is the key control loop: generated content is not accepted just "
        "because the LLM returned text. It must clear the scorer, compliance checks, and retry cap."
    )

    st.header("Key Design Decisions")
    adrs = [
        (
            "ADR-001: Why MCP over direct function calls",
            "Use MCP-style tool servers as the boundary between agents and business data.",
            "MCP makes tools explicit, versionable, and easier to replace when a mock service "
            "becomes a real retailer or catalog API.",
            "It adds a little ceremony compared with direct imports, but the payoff is cleaner "
            "ownership and fewer hidden dependencies.",
        ),
        (
            "ADR-002: Why LangGraph over LangChain chains",
            "Use LangGraph because the pipeline has state, branches, and retry loops.",
            "A chain is fine for straight-line prompting; this system needs conditional routing "
            "from the quality gate back to generation.",
            "LangGraph takes more setup, but it makes failures and state transitions visible.",
        ),
        (
            "ADR-003: Why RAG with two modes",
            "Support lightweight keyword retrieval for portability and vector retrieval for production quality.",
            "Keyword mode is easy to run anywhere. Vector mode with ChromaDB and sentence-transformers "
            "captures semantic matches in retailer guidance.",
            "Vector search needs model weights and RAM, while keyword retrieval is less nuanced.",
        ),
        (
            "ADR-004: Why model-agnostic provider abstraction",
            "Agents depend on an LLMProvider interface, not a specific vendor SDK.",
            "That lets development run on Ollama while production can use Anthropic, OpenAI, or "
            "HuggingFace without rewriting agent logic.",
            "The abstraction hides provider-specific features, so advanced model capabilities need "
            "careful interface design.",
        ),
        (
            "ADR-005: Why the quality gate loop",
            "Generated content must be measured before it is considered usable.",
            "The loop turns vague feedback into targeted repair instructions and gives the model a "
            "bounded chance to improve.",
            "Retries add latency, so the cap protects the user from endless or expensive runs.",
        ),
        (
            "ADR-006: Why local-first observability",
            "Persist every run locally while optionally sending traces to LangSmith.",
            "SQLite keeps the demo and developer workflow portable, while LangSmith gives production "
            "teams deeper trace inspection when a real API key is configured.",
            "Local storage is not a warehouse, but it is reliable enough for debugging and demos.",
        ),
    ]
    for title, decision, rationale, tradeoff in adrs:
        with st.expander(title):
            st.markdown(f"**Decision:** {decision}")
            st.markdown(f"**Rationale:** {rationale}")
            st.markdown(f"**Trade-off:** {tradeoff}")

    st.header("Tech Stack")
    tech_stack = [
        ("Orchestration", "LangGraph — explicit state, conditional edges, retry loops"),
        ("Tool Layer", "MCP Servers — versioned, decoupled, production-standard"),
        ("Vector Store", "ChromaDB — local, zero setup, sufficient for doc scale"),
        (
            "Embeddings",
            "sentence-transformers/all-MiniLM-L6-v2 — free, local, no API key",
        ),
        ("RAG Mode (default)", "Vector similarity — production-quality retrieval over ChromaDB"),
        ("RAG Mode (fallback)", "Keyword matching — lightweight mode when embeddings are unavailable"),
        ("LLM (default)", "Ollama/llama3 — local, no API key, free"),
        ("LLM (production)", "Anthropic Claude — best reasoning quality"),
        ("API", "FastAPI — async, auto-docs at /docs"),
        ("Database", "SQLite via aiosqlite — zero infra, portable"),
        ("Observability", "LangSmith + SQLite — full trace + local fallback"),
    ]
    left, right = st.columns([1, 2])
    left.markdown("**Component**")
    right.markdown("**Technology + why chosen**")
    for component, technology in tech_stack:
        left, right = st.columns([1, 2])
        left.markdown(component)
        right.caption(technology)

    st.header("Current Configuration")
    model_name = {
        "anthropic": settings.anthropic_model,
        "ollama": settings.ollama_model,
        "huggingface_local": settings.hf_model,
        "huggingface_api": settings.hf_api_model,
        "openai": settings.openai_model,
    }.get(settings.llm_provider, "unknown")
    rag_mode_label = {
        "keyword": "Keyword matching",
        "vector": "Vector similarity over ChromaDB",
    }.get(settings.rag_mode.lower(), settings.rag_mode)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🧠 **LLM Provider**\n\n{settings.llm_provider}")
        st.info(f"📦 **Model**\n\n{model_name}")
        st.success(f"🎯 **Quality Gate**\n\n{settings.quality_gate_threshold}/100")
    with col2:
        st.info(f"🔎 **RAG Mode**\n\n{rag_mode_label}")
        st.info(f"🧬 **Embedding Model**\n\n{settings.embedding_model}")
        st.warning(f"🔁 **Max Retries**\n\n{settings.quality_gate_max_retries}")
