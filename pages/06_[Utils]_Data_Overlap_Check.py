#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset-overlap candidate screening for EnvGeo-Seawater.

This page identifies possible duplicate records between cited reference
datasets.  It is an audit tool: neither source files nor displayed analysis
data are changed here.
"""

import numpy as np
import pandas as pd
import streamlit as st

import envgeo_assets
import envgeo_user_data
import envgeo_utils


# =============================================================================
# Data-overlap audit page / データセット間重複候補の監査ページ
# =============================================================================
# This page screens possible overlap without changing any source record.
# Candidate tables support provenance review; they do not prove duplication.
# 元データを変えずに候補を確認する監査ページであり、候補表は重複の確定ではない。

# Keep the app-wide page width consistent with the other public pages.  The
# audit tables themselves retain their own scrollable, width-aware rendering.
st.set_page_config(page_title="Data overlap check | EnvGeo-Seawater", layout="centered")

# Bundled-source catalogue / 同梱データセットの監査用カタログ
SOURCE_FILES = {
    "EnvGeo Dataset [ECS–Japan Sea]": "dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx",
    "Around Japan": "dataset/11_AROUND_JAPAN_PUB_20260305.xlsx",
    "NASA GISS global": "dataset/71_GLOBA_NASA_20260226.xlsx",
    "CoralHydro2k global": "dataset/71_GLOBAL_Atwood_et_al_2026_v02.xlsx",
}
SOURCE_PAIRS = {
    "NASA GISS global × CoralHydro2k global": ("NASA GISS global", "CoralHydro2k global"),
    "Around Japan × NASA GISS global": ("Around Japan", "NASA GISS global"),
    "Around Japan × CoralHydro2k global": ("Around Japan", "CoralHydro2k global"),
    # The three EnvGeo core checks are available for completeness, after the three
    # core integrated-dataset comparisons in the selector.
    "EnvGeo Dataset [ECS–Japan Sea] × Around Japan": ("EnvGeo Dataset [ECS–Japan Sea]", "Around Japan"),
    "EnvGeo Dataset [ECS–Japan Sea] × NASA GISS global": ("EnvGeo Dataset [ECS–Japan Sea]", "NASA GISS global"),
    "EnvGeo Dataset [ECS–Japan Sea] × CoralHydro2k global": ("EnvGeo Dataset [ECS–Japan Sea]", "CoralHydro2k global"),
}
UPLOADED_SOURCE_LABEL = "Uploaded data"

# Audit-table and source-row display fields / 監査表・元行表示の項目
AUDIT_DISPLAY_COLUMNS = [
    "Candidate_ID", "Candidate_Class", "Review_Difference_Category", "Difference_Fields", "Candidate_Reason", "Year", "Month",
    "Latitude_Difference_deg", "Longitude_Difference_deg", "Depth_Difference_m",
    "Salinity_Difference", "d18O_Difference",
    "Latitude_Rounding_Allowance_deg", "Longitude_Rounding_Allowance_deg", "Depth_Rounding_Allowance_m",
    "Salinity_Rounding_Allowance", "d18O_Rounding_Allowance",
    "Rounding_Compatible", "One_to_One_Rounding_Match",
    "Coordinate_Depth_Strong_Exceeded", "Salinity_Strong_Exceeded", "d18O_Strong_Exceeded",
    "Salinity_Right_minus_Left", "d18O_Right_minus_Left", "Salinity_d18O_Change_Direction",
    "Left_Source", "Left_Citation", "Left_Citation_Field", "Left_Row",
    "Right_Source", "Right_Citation", "Right_Citation_Field", "Right_Row",
]
RECORD_DISPLAY_COLUMNS = [
    "Dataset", "reference", "reference_full", "Dataset citation", "Publication DOI or URL",
    "Data provenance notes", "Cruise", "Station", "Date", "Year", "Month", "Day",
    "Latitude_degN", "Longitude_degE", "Depth_m", "Temperature_degC", "Salinity", "d18O", "dD",
]
REVIEW_DECISION_COLUMNS = [
    "Candidate_ID", "Decision", "Display_Action", "Reviewer_Note", "Recorded_At_UTC",
    "Left_Source", "Left_Row", "Left_Citation",
    "Right_Source", "Right_Row", "Right_Citation",
]

# The audit output is intended for scientific review, not only for developers.
# Keep this dictionary beside the displayed-column list so every exported field
# has an explanation visible in the app.
AUDIT_COLUMN_GUIDE = [
    ("Candidate_ID", "Unique identifier for the candidate pair, built from the left/right source names and source-row positions.", "候補対の一意な識別子。左・右のデータセット名と元表の行位置から作成します。"),
    ("Candidate_Class", "Strong candidate: all strict thresholds are met. Review candidate: the broader review thresholds are met but one or more strict thresholds are exceeded. Neither confirms a duplicate.", "Strong candidate は全ての厳しい閾値内、Review candidate は広い確認用閾値内だが厳しい閾値を一部超えた候補です。いずれも重複の確定ではありません。"),
    ("Review_Difference_Category", "For Review candidates, the primary strict-threshold difference: coordinate/depth only, salinity, δ18O, or both salinity and δ18O. Coordinate/depth exceedance is also retained separately.", "Review候補で主に何が厳しい閾値を超えたか。座標・深度のみ／塩分／δ18O／塩分とδ18Oの両方、の4分類です。座標・深度の超過は別列にも示します。"),
    ("Difference_Fields", "All fields exceeding their strict Strong threshold, for example `Latitude; Salinity`.", "厳しい Strong 閾値を超えた全項目の一覧。例：Latitude; Salinity。"),
    ("Candidate_Reason", "Common screening rationale: the pair has the same valid year and month and is within the broader Review criteria.", "候補抽出の共通根拠。同じ有効な年・月で、広い Review 閾値内に入ったことを示します。"),
    ("Year", "Sampling year used for comparison. Only rows with valid matching Year and Month values are compared.", "比較に用いた採水年。YearとMonthの両方が有効で一致する行だけを比較します。"),
    ("Month", "Sampling month used for comparison. A sampling day is not required for the initial screen.", "比較に用いた採水月。初期スクリーニングでは採水日は必須ではありません。"),
    ("Latitude_Difference_deg", "Absolute latitude difference in degrees. A smaller value means the recorded locations are closer.", "緯度の絶対差（度）。小さいほど近い記録です。"),
    ("Longitude_Difference_deg", "Absolute longitude difference in degrees, calculated as the shortest difference across the date line when necessary.", "経度の絶対差（度）。日付変更線をまたぐ場合も最短差で計算します。"),
    ("Depth_Difference_m", "Absolute depth difference in metres. Sampling-depth conventions, CTD representative depth, and rounding can all create a difference.", "深度の絶対差（m）。採水深度・CTD代表深度・丸め方の違いでも差が生じます。"),
    ("Salinity_Difference", "Absolute salinity difference. It gives the size of the difference, not which record is higher.", "塩分の絶対差。どちらが高いかを示さず、差の大きさだけを示します。"),
    ("d18O_Difference", "Absolute δ18O difference in ‰. It gives the size of the difference, not which record is higher.", "δ18O の絶対差（‰）。どちらが高いかを示さず、差の大きさだけを示します。"),
    ("Latitude_Rounding_Allowance_deg", "Estimated combined rounding allowance for latitude, inferred from the displayed decimal precision of both records. It is a screening estimate, not analytical uncertainty.", "左右の表示桁数から推定した緯度の丸め許容幅の合計です。スクリーニング用の推定であり、分析誤差ではありません。"),
    ("Longitude_Rounding_Allowance_deg", "Estimated combined rounding allowance for longitude, inferred from the displayed decimal precision of both records. It is a screening estimate, not analytical uncertainty.", "左右の表示桁数から推定した経度の丸め許容幅の合計です。スクリーニング用の推定であり、分析誤差ではありません。"),
    ("Depth_Rounding_Allowance_m", "Estimated combined rounding allowance for depth, inferred from the displayed decimal precision of both records. It is a screening estimate, not analytical uncertainty.", "左右の表示桁数から推定した深度の丸め許容幅の合計です。スクリーニング用の推定であり、分析誤差ではありません。"),
    ("Salinity_Rounding_Allowance", "Estimated combined rounding allowance for salinity, inferred from the displayed decimal precision of both records. It is a screening estimate, not analytical uncertainty.", "左右の表示桁数から推定した塩分の丸め許容幅の合計です。スクリーニング用の推定であり、分析誤差ではありません。"),
    ("d18O_Rounding_Allowance", "Estimated combined rounding allowance for δ18O, inferred from the displayed decimal precision of both records. It is a screening estimate, not analytical uncertainty.", "左右の表示桁数から推定したδ18Oの丸め許容幅の合計です。スクリーニング用の推定であり、分析誤差ではありません。"),
    ("Rounding_Compatible", "True when every compared difference can be explained by the estimated combined rounding allowance. It does not confirm that the records are duplicates.", "全ての比較差が推定丸め幅で説明できる場合にTrueです。重複の確定ではありません。"),
    ("One_to_One_Rounding_Match", "A provisional one-to-one link selected from Strong, rounding-compatible candidates by the smallest combined normalized difference. It is the candidate list for source review, not an exclusion list.", "Strongかつ丸め整合の候補から、正規化差の合計が最小となる一対一対応として暫定選択したリンクです。一次資料確認用の候補であり、除外リストではありません。"),
    ("Coordinate_Depth_Strong_Exceeded", "Whether at least one of latitude, longitude, or depth exceeds its Strong threshold. This is independent of the salinity/δ18O category.", "緯度・経度・深度のいずれかが Strong 閾値を超えたか。塩分・δ18Oの分類とは独立です。"),
    ("Salinity_Strong_Exceeded", "Whether the salinity difference exceeds its Strong threshold.", "塩分差が Strong 閾値を超えたか。"),
    ("d18O_Strong_Exceeded", "Whether the δ18O difference exceeds its Strong threshold.", "δ18O差が Strong 閾値を超えたか。"),
    ("Salinity_Right_minus_Left", "Salinity of the right record minus salinity of the left record. Positive means the right record is higher; negative means the left is higher.", "右側レコードの塩分 − 左側レコードの塩分。正なら右が高く、負なら左が高いことを示します。"),
    ("d18O_Right_minus_Left", "δ18O of the right record minus δ18O of the left record, in ‰. Positive means the right record is higher; negative means the left is higher.", "右側レコードのδ18O − 左側レコードのδ18O（‰）。正なら右が高く、負なら左が高いことを示します。"),
    ("Salinity_d18O_Change_Direction", "A review cue showing whether the signed salinity and δ18O changes have the same or opposite signs. It does not determine mixing, an error, or a duplicate.", "左右差の符号が同じか逆かを示す補助情報。混合・誤記・重複を自動判定するものではなく、一次資料確認の目印です。"),
    ("Left_Source", "Name of the left dataset in the selected pair. It defines the left side of signed-difference fields.", "選んだデータセット対の左側の名前。符号付き差の左側を定義します。"),
    ("Left_Citation", "Citation displayed for the left source row. The app uses the first available value in this order: `reference`, `reference_full`, then `Dataset citation`.", "左側の元データ行に対して表示する引用情報。`reference`、`reference_full`、`Dataset citation`の順で、最初に値があるものを使います。"),
    ("Left_Citation_Field", "Name of the left source column that supplied `Left_Citation`, making any fallback visible and auditable.", "`Left_Citation`が元データのどの列から得られたかを示します。fallbackを使った場合も追跡できます。"),
    ("Left_Row", "Source-row identifier for the left record. Use it for the side-by-side viewer and source-material checks.", "左側レコードの元データ表における行識別子。左右比較と元資料の照合に使います。"),
    ("Right_Source", "Name of the right dataset in the selected pair. It defines the right side of signed-difference fields.", "選んだデータセット対の右側の名前。符号付き差の右側を定義します。"),
    ("Right_Citation", "Citation displayed for the right source row. The app uses the first available value in this order: `reference`, `reference_full`, then `Dataset citation`.", "右側の元データ行に対して表示する引用情報。`reference`、`reference_full`、`Dataset citation`の順で、最初に値があるものを使います。"),
    ("Right_Citation_Field", "Name of the right source column that supplied `Right_Citation`, making any fallback visible and auditable.", "`Right_Citation`が元データのどの列から得られたかを示します。fallbackを使った場合も追跡できます。"),
    ("Right_Row", "Source-row identifier for the right record. Use it for the side-by-side viewer and source-material checks.", "右側レコードの元データ表における行識別子。左右比較と元資料の照合に使います。"),
]


# =============================================================================
# Source loading and upload handling / 元データ読込とアップロード処理
# =============================================================================
# Bundled workbooks are read into temporary screening copies. Uploaded data
# stay in browser-session memory and are never written to disk.
# 同梱workbookは監査用の一時コピーとして読み、アップロードデータはセッション内だけに保持する。
@st.cache_data(show_spinner=False)
def load_overlap_source(source_name: str) -> pd.DataFrame:
    """Load a cited source into a numeric working copy for overlap screening."""
    path = envgeo_assets.asset_path(SOURCE_FILES[source_name])
    source = pd.read_excel(path).mask(lambda frame: frame.eq("**"), np.nan)
    source["Dataset"] = source_name
    for column in ("Latitude_degN", "Longitude_degE", "Year", "Month", "Depth_m", "Salinity", "d18O"):
        if column in source.columns:
            source[column] = envgeo_utils.coerce_numeric_values(source[column])
    return source


def resolve_overlap_source(source_name: str, uploaded_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a screening copy of either a bundled source or session upload.

    The browser upload is deliberately session-only.  This helper never writes
    it to disk and never changes its values or column assignments.
    """
    if source_name == UPLOADED_SOURCE_LABEL:
        return uploaded_frame.copy()
    return load_overlap_source(source_name)


def upload_screen_signature(uploaded_frame: pd.DataFrame):
    """Return a session-local fingerprint so stale upload audits are not shown."""
    if uploaded_frame is None or uploaded_frame.empty:
        return None
    value_hash = pd.util.hash_pandas_object(
        uploaded_frame.astype(str), index=True
    ).sum()
    return (
        envgeo_utils.get_uploaded_filename(),
        tuple(map(str, uploaded_frame.columns)),
        len(uploaded_frame),
        int(value_hash),
    )


# =============================================================================
# Screening criteria controls / スクリーニング閾値の操作
# =============================================================================
# Strong and Review criteria are visible and adjustable only on this audit
# page. Review must remain equal to or broader than Strong.
# Strong・Review閾値はこの監査ページだけで調整し、ReviewはStrong以上に保つ。
def build_criteria_controls() -> dict:
    """Render explicit, auditable threshold controls in the sidebar."""
    defaults = envgeo_utils.OVERLAP_DEFAULT_CRITERIA
    # Match the bordered sidebar-control blocks used by the analysis pages.
    with st.sidebar.container(border=True):
        st.subheader("Overlap criteria")
        st.caption("Only rows with the same valid Year and Month are compared.")
        strong_latitude = st.number_input("Strong: latitude difference (°)", 0.0, 5.0, float(defaults["max_latitude_difference_deg"]), 0.01, format="%.2f")
        strong_longitude = st.number_input("Strong: longitude difference (°)", 0.0, 5.0, float(defaults["max_longitude_difference_deg"]), 0.01, format="%.2f")
        strong_depth = st.number_input("Strong: depth difference (m)", 1.0, 500.0, float(defaults["max_depth_difference_m"]), 1.0)
        strong_salinity = st.number_input("Strong: salinity difference", 0.0, 5.0, float(defaults["max_salinity_difference"]), 0.01, format="%.2f")
        strong_d18o = st.number_input("Strong: δ18O difference (‰)", 0.0, 5.0, float(defaults["max_d18o_difference"]), 0.01, format="%.2f")
        st.divider()
        st.caption("The review envelope must be equal to or broader than the strong criteria.")
        review_latitude = st.number_input("Review: latitude difference (°)", strong_latitude, 10.0, max(strong_latitude, float(defaults["review_max_latitude_difference_deg"])), 0.01, format="%.2f")
        review_longitude = st.number_input("Review: longitude difference (°)", strong_longitude, 10.0, max(strong_longitude, float(defaults["review_max_longitude_difference_deg"])), 0.01, format="%.2f")
        review_depth = st.number_input("Review: depth difference (m)", 1.0, 1000.0, max(strong_depth, float(defaults["review_max_depth_difference_m"])), 1.0)
        review_salinity = st.number_input("Review: salinity difference", strong_salinity, 10.0, max(strong_salinity, float(defaults["review_max_salinity_difference"])), 0.01, format="%.2f")
        review_d18o = st.number_input("Review: δ18O difference (‰)", strong_d18o, 10.0, max(strong_d18o, float(defaults["review_max_d18o_difference"])), 0.01, format="%.2f")
    return {
        "max_latitude_difference_deg": strong_latitude,
        "max_longitude_difference_deg": strong_longitude,
        "max_depth_difference_m": strong_depth,
        "max_salinity_difference": strong_salinity,
        "max_d18o_difference": strong_d18o,
        "review_max_latitude_difference_deg": review_latitude,
        "review_max_longitude_difference_deg": review_longitude,
        "review_max_depth_difference_m": review_depth,
        "review_max_salinity_difference": review_salinity,
        "review_max_d18o_difference": review_d18o,
    }


# =============================================================================
# Candidate-row inspection / 候補となった元行の確認表示
# =============================================================================
# These helpers present the two original rows side by side; they do not alter
# source values or decide whether a candidate is a duplicate.
# 左右の元行を並べて表示する補助処理であり、値の変更や重複の確定は行わない。
def source_record_for_audit_row(source: pd.DataFrame, row_id):
    """Return one source row addressed by an audit-table index value."""
    matches = source.loc[source.index == row_id]
    if matches.empty and isinstance(row_id, (float, np.floating)) and float(row_id).is_integer():
        matches = source.loc[source.index == int(row_id)]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype="object")


def comparison_table(audit_row, left_frame, right_frame) -> pd.DataFrame:
    """Lay out the original values of one candidate pair side by side."""
    left_record = source_record_for_audit_row(left_frame, audit_row["Left_Row"])
    right_record = source_record_for_audit_row(right_frame, audit_row["Right_Row"])
    fields = [field for field in RECORD_DISPLAY_COLUMNS if field in left_frame.columns or field in right_frame.columns]
    return pd.DataFrame({
        "Field": fields,
        str(audit_row["Left_Source"]): [left_record.get(field, np.nan) for field in fields],
        str(audit_row["Right_Source"]): [right_record.get(field, np.nan) for field in fields],
    })


def render_candidate_inspector(
    audit_subset: pd.DataFrame,
    left_source: str,
    right_source: str,
    key: str,
    uploaded_frame: pd.DataFrame,
) -> None:
    """Render a source-record comparison limited to one candidate class."""
    if audit_subset.empty:
        return
    with st.expander(f"Inspect one {key} candidate pair", expanded=False):
        selected_id = st.selectbox("Candidate ID", audit_subset["Candidate_ID"].tolist(), key=f"{key}_candidate_id")
        selected = audit_subset.loc[audit_subset["Candidate_ID"] == selected_id].iloc[0]
        selected_left = resolve_overlap_source(left_source, uploaded_frame)
        selected_right = resolve_overlap_source(right_source, uploaded_frame)
        st.dataframe(
            envgeo_utils.arrow_display_dataframe(comparison_table(selected, selected_left, selected_right)),
            hide_index=True,
            **envgeo_utils.stretch_width_kwargs(st.dataframe),
        )
        st.caption(
            "Compare source, cruise/station, date, coordinates and profile values before marking any pair "
            "as a confirmed duplicate in a future review workflow."
        )


# =============================================================================
# Optional review notes / 任意のレビュー記録
# =============================================================================
# Decisions remain only in the current session until exported as a separate
# CSV. This optional record is never used to modify source workbooks.
# 判断はCSV出力まで現在のセッションにのみ保持し、元workbookは変更しない。
def render_provisional_decision_tracker(audit_subset: pd.DataFrame) -> None:
    """Record source-review decisions in session state and export them as CSV.

    Decisions deliberately remain separate from source records and are never
    written to a workbook.  The downloadable CSV is the hand-off artifact for
    a later reviewed manifest of confirmed duplicates.
    """
    if audit_subset.empty:
        return
    decisions = st.session_state.setdefault("overlap_review_decisions", {})
    with st.expander("Record a provisional-candidate decision", expanded=False):
        candidate_id = st.selectbox(
            "Candidate ID for review decision",
            audit_subset["Candidate_ID"].tolist(),
            key="provisional_decision_candidate_id",
        )
        selected = audit_subset.loc[audit_subset["Candidate_ID"] == candidate_id].iloc[0]
        prior = decisions.get(candidate_id, {})
        decision_options = ["Pending", "Confirmed duplicate", "Keep both"]
        prior_decision = prior.get("Decision", "Pending")
        decision = st.selectbox(
            "Review decision",
            decision_options,
            index=decision_options.index(prior_decision),
            key=f"provisional_decision_status_{candidate_id}",
        )
        display_action_options = [
            "No display action",
            "Retain left record / hide right record",
            "Retain right record / hide left record",
        ]
        prior_action = prior.get("Display_Action", "No display action")
        display_action = st.selectbox(
            "Display action after confirmation",
            display_action_options,
            index=display_action_options.index(prior_action),
            key=f"provisional_decision_action_{candidate_id}",
            help=(
                "Choose an action only after confirming a duplicate from the source records. "
                "The source workbooks remain unchanged; a later shared display filter will use this field."
            ),
        )
        if decision != "Confirmed duplicate" and display_action != "No display action":
            st.info("A display action is recorded but will not be used unless the decision is Confirmed duplicate.")
        note = st.text_area(
            "Reviewer note (source evidence, reason, or next check)",
            value=prior.get("Reviewer_Note", ""),
            key=f"provisional_decision_note_{candidate_id}",
        )
        if st.button("Save review decision in this session", key="save_provisional_review_decision"):
            decisions[candidate_id] = {
                "Candidate_ID": candidate_id,
                "Decision": decision,
                "Display_Action": display_action,
                "Reviewer_Note": note,
                "Recorded_At_UTC": pd.Timestamp.now(tz="UTC").isoformat(),
                "Left_Source": selected["Left_Source"],
                "Left_Row": selected["Left_Row"],
                "Left_Citation": selected["Left_Citation"],
                "Right_Source": selected["Right_Source"],
                "Right_Row": selected["Right_Row"],
                "Right_Citation": selected["Right_Citation"],
            }
            st.success("Decision recorded for this browser session.")

    decision_frame = pd.DataFrame(list(decisions.values()), columns=REVIEW_DECISION_COLUMNS)
    if decision_frame.empty:
        st.caption("No provisional-candidate decisions have been recorded in this session.")
        return
    st.subheader(f"Session review decisions ({len(decision_frame):,})")
    st.dataframe(
        envgeo_utils.arrow_display_dataframe(decision_frame),
        hide_index=True,
        **envgeo_utils.stretch_width_kwargs(st.dataframe),
    )
    st.download_button(
        "Download provisional review decisions CSV",
        decision_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="envgeo_seawater_overlap_provisional_review_decisions.csv",
        mime="text/csv",
    )


# =============================================================================
# Audit guide display / 監査表の読み方の表示
# =============================================================================
# English and Japanese guides are deliberately separated into tabs so that
# each explanation remains readable without mixing languages.
# 英語・日本語の説明を別タブに分け、各言語で読みやすく表示する。
def render_audit_guide() -> None:
    """Explain every audit field before users interpret candidate records."""
    class_guide = pd.DataFrame(
        [
            {
                "Class": "Strong candidate",
                "Meaning (English)": (
                    "The pair has the same valid Year and Month and is within every "
                    "current Strong threshold for latitude, longitude, depth, salinity, and δ18O. "
                    "It is the closer match class, but still requires source-level confirmation."
                ),
                "意味（日本語）": (
                    "有効な採水年・月が一致し、緯度・経度・深度・塩分・δ18Oのすべてが、"
                    "現在設定されている Strong 閾値内に入る候補です。より近い一致ですが、"
                    "それでも元資料の確認なしに重複確定にはなりません。"
                ),
            },
            {
                "Class": "Review candidate",
                "Meaning (English)": (
                    "The pair is within every broader Review threshold, but exceeds at least one "
                    "Strong threshold. It is retained for inspection because rounding, metadata, "
                    "reprocessing, or genuinely nearby distinct samples can produce this pattern. "
                    "Do not remove either record automatically."
                ),
                "意味（日本語）": (
                    "すべての広い Review 閾値内には入る一方、少なくとも一つの Strong 閾値を"
                    "超える候補です。丸め、メタデータの違い、再処理、近接した別試料でも起こり得るため、"
                    "確認用に残します。どちらの記録も自動的に除外しません。"
                ),
            },
        ]
    )
    guide_frame = pd.DataFrame(
        AUDIT_COLUMN_GUIDE,
        columns=["Column", "Meaning (English)", "意味（日本語）"],
    )

    english_tab, japanese_tab = st.tabs(["English", "日本語"])
    with english_tab:
        st.caption("Cross-dataset overlap candidate check (audit)")
        st.markdown(
            "**This tool identifies possible duplicate records only.** It does not alter source "
            "workbooks, does not confirm duplicates, and does not remove records from analyses."
        )
        st.markdown(
            "Use this page to create an auditable candidate list before reviewing source "
            "provenance, cruises, stations, and measurement metadata."
        )
        st.markdown(
            "A session-only uploaded dataset can also be screened against a selected bundled "
            "EnvGeo dataset for possible overlapping records. To compare it, use "
            "**Uploaded data overlay** in the sidebar and provide the required columns."
        )
        st.markdown(
            "On analytical pages, **Data filtering → Duplicate-candidate display** lets you choose "
            "whether candidate records are shown or hidden. This is a reversible display setting; "
            "source records remain unchanged."
        )
        with st.expander("How to read the audit table", expanded=False):
            st.markdown(
                "**Important:** This is an audit table of possible overlaps. No row is confirmed "
                "as a duplicate, an error, or an exclusion target by this table alone."
            )
            st.markdown(
                "**Left / Right:** The order shown in the dataset-pair selector. "
                "`Right_minus_Left` always means right record minus left record."
            )
            st.subheader("Strong and Review candidates")
            st.dataframe(class_guide[["Class", "Meaning (English)"]], hide_index=True, **envgeo_utils.stretch_width_kwargs(st.dataframe))
            st.caption("The two classes are disjoint: each candidate pair appears in either Strong or Review, never both.")
            st.dataframe(guide_frame[["Column", "Meaning (English)"]], hide_index=True, **envgeo_utils.stretch_width_kwargs(st.dataframe))
            st.download_button("Download English audit-column guide CSV", guide_frame[["Column", "Meaning (English)"]].to_csv(index=False).encode("utf-8-sig"), file_name="envgeo_seawater_overlap_audit_column_guide_en.csv", mime="text/csv")

    with japanese_tab:
        st.caption("データセット間の重複候補チェック（監査用）")
        st.markdown(
            "**このツールは重複データ候補を抽出するだけです。** 元のworkbookを変更せず、"
            "重複を確定せず、解析から記録を除外しません。"
        )
        st.markdown(
            "元データの出典、航海、観測点、測定メタデータを確認する前の、"
            "監査可能な候補一覧を作成するために使います。"
        )
        st.markdown(
            "ブラウザの現在のセッションでアップロードしたユーザーデータについても、"
            "選択したEnvGeo同梱データセットとの重複候補を確認できます。比較するには、"
            "サイドバーの**Uploaded data overlay**を使用し、必要な列を指定してください。"
        )
        st.markdown(
            "各探索ページでは、**Data filtering → Duplicate-candidate display**で、"
            "候補データを表示するか非表示にするかを選べます。これは可逆的な表示設定であり、"
            "元データは変更しません。"
        )
        with st.expander("監査表の読み方", expanded=False):
            st.markdown("**重要：** この表は重複候補を示す監査用の一覧です。どの候補も、この表だけで重複・誤り・除外対象とは決まりません。")
            st.markdown("**左／右：** データセット対のプルダウンに表示された順番です。`Right_minus_Left` は常に「右側 − 左側」を表します。")
            st.subheader("Strong・Review候補の違い")
            st.dataframe(class_guide[["Class", "意味（日本語）"]], hide_index=True, **envgeo_utils.stretch_width_kwargs(st.dataframe))
            st.caption("Strong と Review は重複しない分類であり、各候補対は必ずどちらか一方にだけ表示されます。")
            st.dataframe(guide_frame[["Column", "意味（日本語）"]], hide_index=True, **envgeo_utils.stretch_width_kwargs(st.dataframe))
            st.download_button("監査列ガイドCSVをダウンロード（日本語）", guide_frame[["Column", "意味（日本語）"]].to_csv(index=False).encode("utf-8-sig"), file_name="envgeo_seawater_overlap_audit_column_guide_ja.csv", mime="text/csv")


# =============================================================================
# Page controls and audit execution / ページ操作と監査の実行
# =============================================================================
# A result is kept only in session state. Users must rerun the screen after
# changing the selected pair, criteria, or uploaded data.
# 結果はセッション内だけに保持し、条件・比較対・アップロード変更後は再実行する。
uploaded_df = envgeo_user_data.render_upload_panel(
    "overlap_check",
    "Overlap screening requires Year, Month, latitude, longitude, depth, salinity, and δ18O.",
)
uploaded_df = envgeo_user_data.render_column_controls(
    uploaded_df,
    {
        "Year": "Year",
        "Month": "Month",
        "Latitude": "Latitude_degN",
        "Longitude": "Longitude_degE",
        "Depth": "Depth_m",
        "Salinity": "Salinity",
        "δ18O": "d18O",
    },
    "overlap_check",
)
criteria = build_criteria_controls()
current_upload_signature = upload_screen_signature(uploaded_df)
st.title("Data overlap check")
render_audit_guide()

pair_options = dict(SOURCE_PAIRS)
if not uploaded_df.empty:
    for reference_source in SOURCE_FILES:
        pair_options[f"Uploaded data × {reference_source}"] = (
            UPLOADED_SOURCE_LABEL,
            reference_source,
        )
    filename = envgeo_utils.get_uploaded_filename() or "current session"
    st.caption(
        f"Uploaded-data comparisons use {len(uploaded_df):,} session-only rows from `{filename}`. "
        "Only this one uploaded table is compared; two uploaded tables are not compared on this page."
    )

pair_label = st.selectbox("Dataset pair", list(pair_options), index=0)
left_source, right_source = pair_options[pair_label]
try:
    criteria_note = envgeo_utils.overlap_criteria_text(criteria)
except ValueError as exc:
    st.error(f"Review criteria must be equal to or broader than Strong criteria: {exc}")
    st.stop()
st.caption(criteria_note)

if st.button("Run overlap screen", type="primary"):
    try:
        with st.spinner("Loading selected data and screening candidates…"):
            left_frame = resolve_overlap_source(left_source, uploaded_df)
            right_frame = resolve_overlap_source(right_source, uploaded_df)
            audit = envgeo_utils.screen_dataset_pair_for_overlaps(
                left_frame, right_frame, left_source, right_source, criteria
            )
        st.session_state["overlap_audit"] = audit
        st.session_state["overlap_audit_pair"] = pair_label
        st.session_state["overlap_audit_criteria"] = criteria
        st.session_state["overlap_audit_upload_signature"] = current_upload_signature
        st.session_state["overlap_audit_source_sizes"] = (len(left_frame), len(right_frame))
    except KeyError as exc:
        st.error(
            "The selected table is missing one or more columns required for overlap screening: "
            f"{exc}. Use **Uploaded data columns** in the sidebar to assign them."
        )


# =============================================================================
# Audit results and downloads / 監査結果とCSV出力
# =============================================================================
# Result tables distinguish provisional one-to-one, Strong, and Review
# candidates. They remain candidate evidence, not confirmation of duplication.
# 暫定的一対一・Strong・Reviewを分けて表示し、いずれも重複確定ではない。
audit = st.session_state.get("overlap_audit")
if isinstance(audit, pd.DataFrame):
    if st.session_state.get("overlap_audit_pair") != pair_label:
        st.info("Change detected. Click **Run overlap screen** to evaluate the newly selected dataset pair.")
    elif st.session_state.get("overlap_audit_criteria") != criteria:
        st.info("Criteria changed. Click **Run overlap screen** to update the candidate audit.")
    elif st.session_state.get("overlap_audit_upload_signature") != current_upload_signature:
        st.info("Uploaded data changed. Click **Run overlap screen** to update the candidate audit.")
    elif audit.empty:
        st.success("No candidates were found within the current review criteria.")
    else:
        strong_count = int((audit["Candidate_Class"] == "Strong candidate").sum())
        review_count = int((audit["Candidate_Class"] == "Review candidate").sum())
        left_row_count = int(audit["Left_Row"].nunique())
        right_row_count = int(audit["Right_Row"].nunique())
        source_sizes = st.session_state.get("overlap_audit_source_sizes")
        if source_sizes is not None:
            st.caption("Source rows loaded for this screen")
            source_left, source_right = st.columns(2)
            source_left.metric(f"{left_source} total rows", int(source_sizes[0]))
            source_right.metric(f"{right_source} total rows", int(source_sizes[1]))
        first, second, third, fourth, fifth = st.columns(5)
        first.metric("Candidate pairs", len(audit))
        second.metric("Strong candidates", strong_count)
        third.metric("Review candidates", review_count)
        fourth.metric(f"Unique {left_source} rows", left_row_count)
        fifth.metric(f"Unique {right_source} rows", right_row_count)
        st.caption(
            "Candidate pairs count left/right row combinations, so it can exceed the number of "
            "unique source rows when one record has multiple nearby matches."
        )
        strong_audit = audit.loc[audit["Candidate_Class"] == "Strong candidate", AUDIT_DISPLAY_COLUMNS]
        review_audit = audit.loc[audit["Candidate_Class"] == "Review candidate", AUDIT_DISPLAY_COLUMNS]
        provisional_matches = audit.loc[audit["One_to_One_Rounding_Match"], AUDIT_DISPLAY_COLUMNS]
        st.subheader(f"Provisional one-to-one rounding-compatible matches ({len(provisional_matches):,})")
        st.caption(
            "A stricter subset of Strong candidates: every difference is compatible with the "
            "decimal precision inferred from the imported numeric values, and each source record is used once at most. "
            "Review the original sources before treating a pair as a confirmed duplicate or excluding it."
        )
        st.dataframe(
            envgeo_utils.arrow_display_dataframe(provisional_matches),
            hide_index=True,
            **envgeo_utils.stretch_width_kwargs(st.dataframe),
        )
        st.download_button(
            "Download provisional one-to-one candidate CSV",
            provisional_matches.to_csv(index=False).encode("utf-8-sig"),
            file_name="envgeo_seawater_overlap_provisional_one_to_one_candidates.csv",
            mime="text/csv",
        )
        render_candidate_inspector(
            provisional_matches,
            left_source,
            right_source,
            "Provisional one-to-one",
            uploaded_df,
        )
        render_provisional_decision_tracker(provisional_matches)
        st.subheader(f"Strong candidates ({strong_count:,})")
        st.caption("All strong criteria are met. These are still candidates, not confirmed duplicates.")
        st.dataframe(
            envgeo_utils.arrow_display_dataframe(strong_audit),
            hide_index=True,
            **envgeo_utils.stretch_width_kwargs(st.dataframe),
        )
        st.download_button(
            "Download Strong candidate audit CSV",
            strong_audit.to_csv(index=False).encode("utf-8-sig"),
            file_name="envgeo_seawater_overlap_strong_candidates.csv",
            mime="text/csv",
        )
        render_candidate_inspector(strong_audit, left_source, right_source, "Strong", uploaded_df)
        st.subheader(f"Review candidates ({review_count:,})")
        st.caption(
            "Only the broader review criteria are met. The category is based on salinity and δ18O "
            "differences; the coordinate/depth flag is shown separately. Inspect source metadata before interpretation."
        )
        review_categories = (
            review_audit["Review_Difference_Category"]
            .value_counts()
            .rename_axis("Review difference category")
            .reset_index(name="Candidate pairs")
        )
        st.dataframe(review_categories, hide_index=True, **envgeo_utils.stretch_width_kwargs(st.dataframe))
        st.caption(
            "‘Opposite signed changes’ is a review cue only. It is not, by itself, evidence of an "
            "impossible water-mass relationship or a confirmed duplicate."
        )
        st.dataframe(
            envgeo_utils.arrow_display_dataframe(review_audit),
            hide_index=True,
            **envgeo_utils.stretch_width_kwargs(st.dataframe),
        )
        st.download_button(
            "Download Review candidate audit CSV",
            review_audit.to_csv(index=False).encode("utf-8-sig"),
            file_name="envgeo_seawater_overlap_review_candidates.csv",
            mime="text/csv",
        )
        render_candidate_inspector(review_audit, left_source, right_source, "Review", uploaded_df)
        st.download_button(
            "Download all candidate audit CSV",
            audit.to_csv(index=False).encode("utf-8-sig"),
            file_name="envgeo_seawater_overlap_candidate_audit.csv",
            mime="text/csv",
        )
        st.caption(
            "Rows without usable Year or Month are not compared. Review candidates are not "
            "duplicates by definition; retain both records until provenance review is complete."
        )
else:
    st.info("Choose a dataset pair and run the screen. The result is kept only in this session.")
