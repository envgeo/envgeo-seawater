# Release・Zenodoメタデータ下書き

[English version](release_metadata_draft.md)

これは最終GitHub ReleaseとZenodo recordへ転記するための下書きである。release recordそのものではない。
tag日、clean buildしたwheelのchecksum、Zenodoのversion DOIは、review済みstable commitをtagしてarchiveを
作成した後にだけ確定する。

## 基本メタデータ

| 項目 | 下書き値 |
|---|---|
| Title | EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data |
| Release label | EnvGeo-Seawater v1.3.4 |
| Version | 1.3.4 |
| Author | Toyoho Ishimura |
| ORCID | <https://orcid.org/0000-0001-9708-3743> |
| Affiliation | Graduate School of Human and Environmental Studies, Kyoto University, Japan |
| License | MIT |
| Source repository | <https://github.com/envgeo/seawater_map> |
| Release date | 2026-10-03 |
| Release commit | `948b384455480f06b7a9b6b0a7a3e53af7135e35` |
| Version DOI | <https://doi.org/10.5281/zenodo.23117784> |
| Concept DOI（全version） | <https://doi.org/10.5281/zenodo.23117783> |
| Zenodo record | <https://zenodo.org/records/23117784> |

## 短い説明文

EnvGeo-Seawaterは、海水安定同位体と水文データを探索するためのPython／Streamlitによる
インタラクティブなplatformである。NASA GISSおよびPAGES CoralHydro2kの比較用dataset、地域参照data、
EnvGeo Dataset [ECS–Japan Sea]中核コレクション（主要出典：Kodama et al. (2024)）を含む、約50,000件の引用付きrecordを統合する。地図、
2D–4D可視化、塩分–同位体関係、T–S diagram、深度profile、鉛直section、利用者dataとのsession限定比較を
提供する。

## Keywords

`seawater isotopes`；`oceanography`；`hydrography`；`stable isotopes`；`data visualization`；
`Streamlit`；`Python`

## 安定版の公開範囲

安定版public repositoryにはPage 03、04、05、31、32、34、35、37、53、および履歴archiveのPage 80を含める。
開発用Page 90・91とローカル診断Page 99は、安定版Release、wheel、GitHub Release、Zenodo archiveから除外する。

## 最終確定の手順

1. 最終commitを作り、GitHub Actions CIの結果を確認する。
2. そのstable commitに`v1.3.4` tagを付ける。
3. tagのclean checkoutからwheelを作り、SHA-256を記録する。
4. 上記title・説明文を使い、tagからGitHub Releaseを作る。
5. 対応するZenodo archiveを公開し、version DOIを`CITATION.cff`、READMEの引用文、この記録へ追記する。

引用時には、EnvGeo-Seawater **および** 解析で使った各元データ提供者を引用する。
