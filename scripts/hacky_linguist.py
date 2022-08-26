import yaml
import json

COMMON_LANGUAGES = {'javascript', 'python', 'java', 'php', 'ruby', 'c++', 'go', 'c', 'c#', 'typescript', 'shell', 'swift', 'scala', 'objective-c', 'rust', 'coffeescript', 'haskell', 'perl', 'lua', 'clojure'}

def get_extension_to_language_map(languages_yaml_file, verbose=False):
    # this is not an exhaustive list, but it covered the cases that showed up when running this without the override check
    # 20 most popular languages in quarter 3 of 2016 (roughly when Google Code shut down, by https://madnight.github.io/githut/#/pull_requests/2016/3
    with open(languages_yaml_file, "r") as f:
        languages = yaml.load(f, Loader=yaml.FullLoader)
    extension_to_language = {}
    for language, language_info in sorted(languages.items()):
        language = language.lower()
        for extension in language_info.get('extensions', []):
            if extension in extension_to_language:
                old_language = extension_to_language[extension]
                if old_language in COMMON_LANGUAGES:
                    if verbose:
                        print(f"{extension}: BLOCK  {old_language} (Common) | {language}")
                    continue
                    # do not update
                elif language in COMMON_LANGUAGES:
                    # update
                    if verbose:
                        print(f"{extension}: UPDATE {old_language} -> {language} (Common)")
                else:
                    # neither one is common; first-come-first-served
                    if verbose:
                        print(f"{extension}: BLOCK  {old_language} (First) | {language}")
                    continue
            extension_to_language[extension] = language
    return extension_to_language

EXTENSION_TO_LANGUAGE = get_extension_to_language_map("languages.yml")

def build_language_extensions(filename='./Programming_Languages_Extensions_filtered.json'):
    lang_exts = []
    with open(filename) as f:
        for i in json.load(f):
            if "extensions" not in i:
                continue
            lang_exts.extend(i["extensions"])
    return lang_exts

# load programming language extensions from json file
LANGUAGE_EXTENSIONS = build_language_extensions()