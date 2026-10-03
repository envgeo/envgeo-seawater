---
layout: default
title: EnvGeo-Seawater 利用ガイド
---

# EnvGeo-Seawater 利用ガイド

**Version 1.3.4** · 海水安定同位体・水文データの対話的な探索

[English](index.html) · [ソースリポジトリ](https://github.com/envgeo/seawater_map)

EnvGeo-Seawaterは、地域・全球の参照データセットとユーザー自身の観測データをあわせて、海水安定同位体と水文データを探索するためのアプリケーションです。本サイトでは、安定版でサポートする操作方法を説明します。

> 本アプリケーションは、対話的な探索と品質確認を目的としています。解析では、EnvGeo-Seawaterと利用した元データ提供者の両方を引用してください。

## 最初の操作

1. アプリケーションのサイドバーから可視化ページを開きます。
2. 参照データセットを選び、共通のデータ絞り込み条件を調整します。
3. **Apply settings** を選択して、表と図を更新します。
4. 地図、T–S空間、深度プロファイルなどを探索します。
5. 対応ページではCSVまたはXLSXをuploadできます。uploadしたデータは、そのsession内だけで扱われます。

![地理的選択の例](assets/images/selection_map.png)

## 操作ガイドを選ぶ

### まずはこちら

- [概要](manual_Japanese/00_overview.html)
- [データの絞り込み](manual_Japanese/01_data_filtering.html)
- [User Data Check & Quick Visualizer](manual_Japanese/05_user_data_check_quick_visualizer.html)

### インタラクティブな探索

- [Interactive 2Dplus Visualizer](manual_Japanese/03_2dplus_visualizer.html)
- [Interactive 3D/4D Visualizer](manual_Japanese/04_3d_4d_visualizer.html)
- [同位体・水文マッピング](manual_Japanese/32_isotope_hydrographic_mapping.html)
- [水温–塩分図](manual_Japanese/34_ts_diagram.html)

![水温–塩分図の例](assets/images/ts_diagram.png)

### 解析ビュー

- [塩分–δ18O 関係](manual_Japanese/31_salinity_d18o.html)
- [カスタムパラメータプロット](manual_Japanese/35_custom_parameter_plot.html)
- [深度プロファイル](manual_Japanese/37_depth_profile.html)
- [鉛直断面可視化](manual_Japanese/53_vertical_section.html)

![3D/4D表示の例](assets/images/4d_d18O.png)

## データ・引用・サポート

- [データ出典と引用情報](https://github.com/envgeo/seawater_map/tree/main/data_text)
- [更新履歴](https://github.com/envgeo/seawater_map/blob/main/data_text/update_log_Japanese.md)
- [オフライン・通信不安定時の動作](offline_operation_log_Japanese.html)
- [テストと対応環境](testing_Japanese.html)
- [安定版の公開範囲](stable_release_publication_notes_Japanese.html)

## 本サイトの対象範囲

本サイトでは、安定版でサポートする10ページの操作を扱います。歴史的な開発記録や除外した実験的ページは、安定版の利用者向け機能としては案内しません。
