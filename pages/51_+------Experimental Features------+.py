import streamlit as st
import os
import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# タイトル
st.title("Beta Version Apps")

# ページリストの取得関数
def get_pages():
    pages_dir = "pages/97_beta"  # ページディレクトリ
    if not os.path.exists(pages_dir):  # ディレクトリが存在しない場合の処理
        st.error(f"Directory '{pages_dir}' was not found.")
        return {}
    files = [f for f in os.listdir(pages_dir) if f.endswith(".py")]  # .pyファイルのみ取得
    if not files:  # ファイルが見つからない場合の処理
        st.error(f"No pages were found in {pages_dir}.")
        return {}
    # pages = {file.split("_", 1)[1].replace(".py", ""): f"{pages_dir}/{file}" for file in files}
    # return pages

    # 番号順に並べ替えるため、split("_", 1)の最初の部分をint型としてソート
    pages = {file.split("_", 1)[1].replace(".py", ""): f"{pages_dir}/{file}" for file in sorted(files, key=lambda x: int(x.split("_", 1)[0]))}
    return pages



# ページの読み込み
pages = get_pages()

# ページが存在する場合のみ続行
if pages:
    # セッションステートを使って選択されたページを保存
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = list(pages.keys())[0]  # デフォルトで最初のページを選択

    # ドロップダウンメニュー（keyを固定し、session_stateで管理）
    selected_page = st.selectbox(
    # selected_page = st.sidebar.selectbox(
        "Select a page",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.selected_page)  # 現在選択されているページをデフォルト選択
    )

    # 選択されたページをセッションステートに保存
    # st.session_state.selected_page = selected_page

    # 選択されたページを実行
    page_path = pages[selected_page]
    try:
        runpy.run_path(str(BASE_DIR / page_path), run_name="__main__")
    except Exception as e:
        st.error(f"An error occurred while loading page '{selected_page}': {e}")
