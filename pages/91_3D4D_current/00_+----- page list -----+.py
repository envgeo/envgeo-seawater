# -*- coding: utf-8 -*-

import streamlit as st


def main():
    # # タイトル
    # st.title('SEAWATER DATA on WEB ')
    # # ヘッダ
    # st.header('powered by streamlit')
    # # 純粋なテキスト
    # st.text('Toyoho Isihimura')
    # # サブレベルヘッダ
    # st.subheader('⬅ Select a sub-menu in the sidebar')
    # # マークダウンテキスト
    # st.markdown('**Markdown is available **')
    # # LaTeX テキスト
    # st.latex(r'\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i')
    # # コードスニペット
    # st.code('print(\'Hello, World!\')')
    # # エラーメッセージ
    # st.error('Error message')
    # # 警告メッセージ
    # st.warning('Warning message')
    # # 情報メッセージ
    # st.info('Information message')
    # # 成功メッセージ
    # st.success('Success message')
    # # 例外の出力
    # st.exception(Exception('Oops!'))
    # # 辞書の出力
    # d = {
    #     'foo': 'bar',
    #     'users': [
    #         'alice',
    #         'bob',
    #     ],
    # }
    # st.json(d)

    #gifアニメを読み込む
    st.subheader('🚢 Cruise Area (2015-2021): Kodama et al. (2024)')
    st.image("data/sites_20230515.gif")
        
    #gifアニメを読み込む
    st.subheader('🚢 Sampling sites by Year: Kodama et al. (2024)')
    st.image("data/year_20230517.gif")


    
if __name__ == '__main__':
    main()
