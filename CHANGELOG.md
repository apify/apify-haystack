# Changelog

All notable changes to this project will be documented in this file.

<!-- git-cliff-unreleased-start -->
## 1.7.1 - **not yet released**

### 🚀 Features

- Add support for Python 3.14 ([#36](https://github.com/apify/apify-haystack/pull/36)) ([fe4abac](https://github.com/apify/apify-haystack/commit/fe4abacc1511b1cbd40d98cd9b5d57ef3b9dcb5a)) by [@vdusek](https://github.com/vdusek)
- [**breaking**] Migrate to apify-client 3.x ([#47](https://github.com/apify/apify-haystack/pull/47)) ([8358f91](https://github.com/apify/apify-haystack/commit/8358f915c9b320206e7c11b08a4e64d28bac5aad)) by [@vdusek](https://github.com/vdusek), closes [#28](https://github.com/apify/apify-haystack/issues/28)
- [**breaking**] Migrate to Haystack 3.x ([#39](https://github.com/apify/apify-haystack/pull/39)) ([783706e](https://github.com/apify/apify-haystack/commit/783706eb07e0365efc7f9831de46115028add91f)) by [@vdusek](https://github.com/vdusek)
- [**breaking**] Create the Apify client in `warm_up` and release it in `close` ([#41](https://github.com/apify/apify-haystack/pull/41)) ([405455c](https://github.com/apify/apify-haystack/commit/405455cba9fcfea6c780c315d5634ee783780029)) by [@vdusek](https://github.com/vdusek)
- [**breaking**] Take the API token as a Haystack `Secret` and support component serialization ([#42](https://github.com/apify/apify-haystack/pull/42)) ([f52c359](https://github.com/apify/apify-haystack/commit/f52c35978a86dd69cd4d478d396b052ee76b0853)) by [@vdusek](https://github.com/vdusek)
- Add async support to the components ([#57](https://github.com/apify/apify-haystack/pull/57)) ([3d15741](https://github.com/apify/apify-haystack/commit/3d15741496cdca24441f93806a3bb2d3d61f0e6c)) by [@vdusek](https://github.com/vdusek)
- Add ApifyWebSearch component ([#58](https://github.com/apify/apify-haystack/pull/58)) ([974f861](https://github.com/apify/apify-haystack/commit/974f8616c1632ccf8bf444e2567534bb1371a936)) by [@vdusek](https://github.com/vdusek)
- Add ApifyActorTool for Haystack Agents ([#59](https://github.com/apify/apify-haystack/pull/59)) ([96cfedc](https://github.com/apify/apify-haystack/commit/96cfedcb798f59b17f1b97e116e697065020745c)) by [@vdusek](https://github.com/vdusek)

### 🐛 Bug Fixes

- Respect input passed to `run` in Actor and task calls ([#33](https://github.com/apify/apify-haystack/pull/33)) ([78a3882](https://github.com/apify/apify-haystack/commit/78a388256a8230d397b135bca4c3d0cbce9c16da)) by [@vdusek](https://github.com/vdusek)

### ⚙️ Miscellaneous Tasks

- [**breaking**] Drop support for Python 3.9 ([#34](https://github.com/apify/apify-haystack/pull/34)) ([d86960d](https://github.com/apify/apify-haystack/commit/d86960dbe3ad794541833680cfd05c3331170c26)) by [@vdusek](https://github.com/vdusek)
- [**breaking**] Drop support for Python 3.10 ([#35](https://github.com/apify/apify-haystack/pull/35)) ([5b33fe5](https://github.com/apify/apify-haystack/commit/5b33fe53dd722d584461909edc2d7aacfdb686d9)) by [@vdusek](https://github.com/vdusek)
- [**breaking**] Drop the unused python-dotenv dependency and move examples out of the package ([#43](https://github.com/apify/apify-haystack/pull/43)) ([9b3de5e](https://github.com/apify/apify-haystack/commit/9b3de5e81babcede7d3287fcacf1189f72d02224)) by [@vdusek](https://github.com/vdusek)


<!-- git-cliff-unreleased-end -->
# Changelog

## [0.1.7](https://github.com/apify/apify-haystack/releases/tag/0.1.7)  (2024-12-09)

🐛 Bug Fixes
- Use CI from crawlee-python to publish to PyPI

## [0.1.6](https://github.com/apify/apify-haystack/releases/tag/0.1.6)  (2024-12-09)

🐛 Bug Fixes
- Fix description in pyproject.toml

## [0.1.5](https://github.com/apify/apify-haystack/releases/tag/0.1.5)  (2024-09-04)

🚀 Features
- Add link to google colab in the README

## [0.1.4](https://github.com/apify/apify-haystack/releases/tag/0.1.4)  (2024-09-04)

🚀 Features
- Allow to add Actor inputs (`run_input`) in the ApifyDatasetFromActorCall.run() method

## [0.1.3](https://github.com/apify/apify-haystack/releases/tag/0.1.3)  (2024-08-04)

🚀 Features
- Update code examples in the documentation

## [0.1.2](https://github.com/apify/apify-haystack/releases/tag/0.1.2)  (2024-08-02)

🐛 Bug Fixes
- Correct examples in the documentation

## [0.1.1](https://github.com/apify/apify-haystack/releases/tag/0.1.1)  (2024-07-31)

🐛 Bug Fixes
- Fixed publication to PyPI

## [0.1.0](https://github.com/apify/apify-haystack/releases/tag/0.1.0)  (2024-07-30)

🚀 Features
- Added support for the Apify-Haystack integration
- Added code examples

## [0.0.1] (2024-07-25)

🚀 Features
- Dummy package `apify-haystack` was released on PyPI.
- Added CI
