"""Design system for the Satellite Change Intelligence dashboard.

Base colors, typography, radii, and chart palettes are defined natively in `.streamlit/config.toml`
(Streamlit's own recommended approach — see the bundled theming reference). This module adds only
what native theming cannot: a small amount of key-scoped CSS for gradients, card hover effects, and
a few typographic treatments (kicker labels, hero banner) the user explicitly asked for, plus a set
of small reusable Python components (hero banner, KPI rows, empty states, info banners, status
badges) so every page composes them consistently instead of duplicating markup.
"""
import streamlit as st

ICONS = {
    "overview": ":material/dashboard:",
    "detect": ":material/compare:",
    "models": ":material/query_stats:",
    "geo": ":material/public:",
    "temporal": ":material/schedule:",
    "diagnostics": ":material/build:",
    "satellite": ":material/satellite_alt:",
    "upload": ":material/upload_file:",
    "region": ":material/category:",
    "severity": ":material/warning:",
    "export": ":material/download:",
    "check": ":material/check_circle:",
    "error": ":material/error:",
    "info": ":material/info:",
    "map": ":material/map:",
    "trend": ":material/trending_up:",
    "layers": ":material/layers:",
}


def inject_theme() -> None:
    """Small, targeted CSS additions — gradients, hover states, and a couple of typographic
    treatments not available via config.toml theming alone. Everything else (colors, fonts,
    radii, borders) comes from `.streamlit/config.toml`."""
    st.html(
        """
        <style>
        /* Kicker: small, uppercase, tracked-out label above section headers */
        .sci-kicker {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #22D3EE;
            margin-bottom: 0.15rem;
        }
        .sci-subtle {
            color: #8B95A7;
            font-size: 0.85rem;
        }

        /* Hero banner: subtle gradient wash behind the main brand header */
        div[class*="st-key-hero-banner"] {
            background: linear-gradient(135deg, rgba(34,211,238,0.10) 0%, rgba(167,139,250,0.08) 60%, rgba(10,14,20,0) 100%);
            border: 1px solid #1B2231;
            border-radius: 16px;
            padding: 0.25rem;
        }

        /* Card hover: any container keyed "card-*" gets a subtle lift + glow on hover */
        div[class*="st-key-card-"] {
            transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
        }
        div[class*="st-key-card-"]:hover {
            border-color: #2DD4EE55;
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.28);
        }

        /* Sidebar brand wordmark */
        .sci-brand {
            font-weight: 700;
            font-size: 1.05rem;
            line-height: 1.25;
            letter-spacing: 0.01em;
            color: #E5E9F0;
        }
        .sci-brand span { color: #22D3EE; }

        /* Compact info banner replacing oversized st.info blocks */
        div[class*="st-key-banner-"] {
            border-radius: 10px;
        }
        </style>
        """
    )


def kicker(text: str) -> None:
    st.markdown(f'<div class="sci-kicker">{text}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, tagline: str | None = None) -> None:
    """The main application brand header — used once, at the top of the Overview page."""
    with st.container(key="hero-banner"):
        with st.container(horizontal_alignment="left"):
            if tagline:
                kicker(tagline)
            st.markdown(f"## {title}")
            st.markdown(f'<div class="sci-subtle">{subtitle}</div>', unsafe_allow_html=True)
        st.space("small")


def section_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    label = f"{icon} {title}" if icon else title
    st.markdown(f"### {label}")
    if subtitle:
        st.caption(subtitle)


def kpi_row(items: list[dict]) -> None:
    """items: list of {"label", "value", "help" (optional), "delta" (optional)}"""
    with st.container(horizontal=True):
        for item in items:
            st.metric(
                item["label"], item["value"],
                delta=item.get("delta"),
                help=item.get("help"),
                border=True,
            )


def info_banner(text: str, icon: str = ICONS["info"], link_label: str | None = None,
                 link_page: str | None = None, key: str = "default") -> None:
    """A single-line, compact information strip — replaces the oversized disclaimer boxes."""
    with st.container(border=True, key=f"banner-{key}"):
        cols = st.columns([0.05, 0.75, 0.20]) if link_label else st.columns([0.05, 0.95])
        cols[0].markdown(icon)
        cols[1].markdown(f'<span class="sci-subtle">{text}</span>', unsafe_allow_html=True)
        if link_label and link_page:
            with cols[2]:
                st.page_link(link_page, label=link_label)


def status_badge(text: str, kind: str = "neutral") -> None:
    color_map = {"success": "green", "warning": "orange", "error": "red", "info": "blue", "neutral": "gray"}
    icon_map = {"success": ":material/check:", "warning": ":material/warning:",
                "error": ":material/error:", "info": ":material/info:", "neutral": None}
    st.badge(text, icon=icon_map.get(kind), color=color_map.get(kind, "gray"))


def empty_state(icon: str, title: str, subtitle: str) -> None:
    with st.container(horizontal_alignment="center", key=f"empty-{hash(title) & 0xffff}"):
        st.markdown(f'<div style="font-size:2.4rem;opacity:0.7;">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.caption(subtitle)


def capability_card(icon: str, title: str, description: str, key: str) -> None:
    with st.container(border=True, key=f"card-{key}"):
        st.markdown(f"{icon}  **{title}**")
        st.caption(description)
