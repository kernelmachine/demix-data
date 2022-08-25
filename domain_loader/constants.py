from pathlib import Path
import os

DATA_DIR = Path(os.environ['DATA_DIR'])





TOKEN_COUNTS_DEMIX = {"1b": {'num_train_tokens': 700_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "cs": {'num_train_tokens': 4_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "reddit": {'num_train_tokens': 25_000_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "reviews": {'num_train_tokens': 2_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "realnews": {'num_train_tokens': 15_000_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "openwebtext": {'num_train_tokens': 6_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "legal": {'num_train_tokens': 10_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "med": {'num_train_tokens': 9_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "gutenberg": {'num_train_tokens': 3_000_000_000,
                              'num_dev_tokens': 10_000_000,
                              'num_test_tokens': 10_000_000},
                "qasper": {'num_train_tokens': 1_000_000,
                              'num_dev_tokens': 1_000_000,
                              'num_test_tokens': 1_000_000},
                "legal_contracts": {"num_train_tokens": 1_500_000,
                                    "num_dev_tokens": 1_000_000,
                                    "num_test_tokens": 1_000_000},
                "cord19": {"num_train_tokens": 60_000_000,
                                    "num_dev_tokens": 10_000_000,
                                    "num_test_tokens": 10_000_000},
                "github": {"num_train_tokens": 200_000_000,
                                    "num_dev_tokens": 10_000_000,
                                    "num_test_tokens": 10_000_000},
                "tweets": {"num_train_tokens": 8_000_000,
                                    "num_dev_tokens": 1_000_000,
                                    "num_test_tokens": 1_000_000},
                "yelp_reviews": {"num_train_tokens": 600_000_000,
                                    "num_dev_tokens": 10_000_000,
                                    "num_test_tokens": 10_000_000},
                "latest_news": {"num_train_tokens":11_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000},
                "ag": {"num_train_tokens": 100_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "imdb": {"num_train_tokens": 800_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "1b_test": {'num_train_tokens': 80_000,
                        'num_dev_tokens': 100_000,
                        'num_test_tokens': 100_000},
                "hyperpartisan_news": {"num_train_tokens": 200_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "chemprot": {"num_train_tokens": 100_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "rct": {"num_train_tokens": 800_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "citation_intent": {"num_train_tokens": 50_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "amazon": {"num_train_tokens": 500_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "acl_papers": {"num_train_tokens": 500_000,
			       "num_dev_tokens": 100_000,
                               "num_test_tokens": 100_000}
                }


TOKEN_COUNTS = {"1b": {'num_train_tokens': 990_000_000,
                        'num_dev_tokens': 1_000_000,
                        'num_test_tokens': 1_000_000},
                "cs": {'num_train_tokens': 4_500_000_000,
                        'num_dev_tokens': 1_000_000,
                        'num_test_tokens': 1_000_000},
                "reddit": {'num_train_tokens': 25_000_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "reviews": {'num_train_tokens': 2_500_000_000,
                        'num_dev_tokens': 10_000_000,
                        'num_test_tokens': 10_000_000},
                "realnews": {'num_train_tokens': 10_000_000_000,
                        'num_dev_tokens': 1_000_000,
                        'num_test_tokens': 1_000_000},
                "openwebtext": {'num_train_tokens': 6_500_000_000,
                        'num_dev_tokens': 1_000_000,
                        'num_test_tokens': 1_000_000},
                "legal": {'num_train_tokens': 10_000_000_000,
                        'num_dev_tokens': 1_000_000,
                        'num_test_tokens': 1_000_000},
                "gutenberg": {'num_train_tokens': 3_000_000_000,
                              'num_dev_tokens': 1_000_000,
                              'num_test_tokens': 1_000_000},
                "qasper": {'num_train_tokens': 1_000_000,
                              'num_dev_tokens': 1_000_000,
                              'num_test_tokens': 1_000_000},
                
                "advice": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
                "discussion": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
                "politics_submissions": {"num_train_tokens": 10_000, "num_dev_tokens": 10_000, "num_test_tokens": 1_000_000},
                "gaming_submissions": {"num_train_tokens": 10_000, "num_dev_tokens": 10_000, "num_test_tokens": 800_000},
                "nba_submissions": {"num_train_tokens": 10_000, "num_dev_tokens": 10_000, "num_test_tokens": 950_000},
                "multilex": {"num_train_tokens": 200_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
                "legal_contracts": {"num_train_tokens": 1_500_000,
                                    "num_dev_tokens": 1_000_000,
                                    "num_test_tokens": 1_000_000},
                "cord19": {"num_train_tokens": 1_200_000_000,
                                    "num_dev_tokens": 1_000_000,
                                    "num_test_tokens": 1_000_000},
                "twitter": {"num_train_tokens": 1_800_000_000,
                                    "num_dev_tokens": 1_000_000,
                                    "num_test_tokens": 1_000_000},
                "yelp_reviews": {"num_train_tokens": 600_000_000,
                                    "num_dev_tokens": 10_000_000,
                                    "num_test_tokens": 10_000_000},
                "latest_news": {"num_train_tokens":11_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000},
                "ag": {"num_train_tokens": 100_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "imdb": {"num_train_tokens": 800_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "1b_test": {'num_train_tokens': 80_000,
                        'num_dev_tokens': 100_000,
                        'num_test_tokens': 100_000},
                "hyperpartisan_news": {"num_train_tokens": 200_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "chemprot": {"num_train_tokens": 100_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "rct": {"num_train_tokens": 800_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "citation_intent": {"num_train_tokens": 50_000,
                            "num_dev_tokens": 10_000,
                            "num_test_tokens": 10_000},
                "amazon": {"num_train_tokens": 500_000,
                            "num_dev_tokens": 100_000,
                            "num_test_tokens": 100_000},
                "acl_papers": {"num_train_tokens": 500_000,
			       "num_dev_tokens": 100_000,
                               "num_test_tokens": 100_000},
                "c4": {"num_train_tokens": 10_000_000_000,
			       "num_dev_tokens": 1_000_000,
                               "num_test_tokens": 1_000_000},
                "Rust": { "num_train_tokens":175_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "GO": { "num_train_tokens":1_900_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "Python": { 
                        "num_train_tokens":2_800_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "C": { 
                        "num_train_tokens":4_800_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "C#": { 
                        "num_train_tokens":2_300_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "github": { "num_train_tokens": 2_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Assembly": { 
                        "num_train_tokens":25_000_000,
                          "num_dev_tokens": 1_000_000,
                          "num_test_tokens": 1_000_000},
                "C++": { 
                        "num_train_tokens":3_700_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Java": { 
                        "num_train_tokens":6_800_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "JavaScript": { 
                        "num_train_tokens":7_200_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "SQL": { 
                        "num_train_tokens":250_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Lua": { 
                        "num_train_tokens":120_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Shell": { 
                        "num_train_tokens":180_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "PowerShell": { 
                        "num_train_tokens":45_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Markdown": { 
                        "num_train_tokens":2_100_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "PHP": { 
                        "num_train_tokens":2_200_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Perl": { 
                        "num_train_tokens":170_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "FORTRAN": { 
                        "num_train_tokens":55_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Haskell": { 
                        "num_train_tokens":190_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "HTML": { 
                        "num_train_tokens":5_200_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Ruby": { 
                        "num_train_tokens":855_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Julia": { 
                        "num_train_tokens":22_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "CSS": { 
                        "num_train_tokens":1_400_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "TeX": { 
                        "num_train_tokens":85_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Art": { 
                        "num_train_tokens":95_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Biology": { 
                        "num_train_tokens":9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Business": { 
                        "num_train_tokens":600_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "na": {
                        "num_train_tokens": 2_400_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                        },
                'uspto': {
                        "num_train_tokens": 230_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
               '_split_0': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_1': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_2': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_3': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_4': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_5': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_6': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                '_split_7': {
                        "num_train_tokens": 9_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                'poetry': { 'num_train_tokens': 15_000_000, 'num_dev_tokens': 1_000_000, 'num_test_tokens': 1_000_000},
                "Chemistry": { 
                        "num_train_tokens":2_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "sports_comments": { 
                        "num_train_tokens":125_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "acl": { 
                        "num_train_tokens":140_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},


                 "gaming_comments": { 
                        "num_train_tokens":170_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "supreme_court": { 
                        "num_train_tokens":120_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "stories": { 
                        "num_train_tokens":7_300_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "dm_mathematics": { 
                        "num_train_tokens":1_500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "2021_newscrawl": { 
                        "num_train_tokens":950_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "ireland_speeches": { 
                        "num_train_tokens":50_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "mags": { 
                        "num_train_tokens":100_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "opensubtitles": { 
                        "num_train_tokens":750_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "enron": { 
                        "num_train_tokens":150_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "paper_reviews": { 
                        "num_train_tokens":10_000,
                        "num_dev_tokens": 10_000,
                        "num_test_tokens": 1_000_000},
                  "bills": { 
                        "num_train_tokens":20_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                  "movie_plots": { 
                        "num_train_tokens":3_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "nonfic": { 
                        "num_train_tokens":60_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "fic": { 
                        "num_train_tokens":230_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "coha_news": { 
                        "num_train_tokens":40_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                 "covidtweets": { 
                        "num_train_tokens":1_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "blogtext": { 
                        "num_train_tokens":130_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},




                "toefl": { 
                        "num_train_tokens":1_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "imdb": { 
                        "num_train_tokens":9_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                    "rct": { 
                        "num_train_tokens":4_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "bookcorpus": { 
                        "num_train_tokens":760_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "hackernews": { 
                        "num_train_tokens":845_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "cs": { 
                        "num_train_tokens":6_450_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Economics": { 
                        "num_train_tokens":1_500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Engineering": { 
                        "num_train_tokens":1_300_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "env_sci": { 
                        "num_train_tokens":250_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Geography": { 
                        "num_train_tokens":350_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Geology": { 
                        "num_train_tokens":985_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "History": { 
                        "num_train_tokens":175_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "materials": { 
                        "num_train_tokens":1_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Mathematics": { 
                        "num_train_tokens":4_500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Medicine": { 
                        "num_train_tokens":10_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Philosophy": { 
                        "num_train_tokens":150_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Physics": { 
                        "num_train_tokens":5_250_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "polisci": {            
                        "num_train_tokens":450_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Psychology": { 
                        "num_train_tokens":2_250_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Sociology": { 
                        "num_train_tokens":950_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "wikipedia": { 
                        "num_train_tokens":2_500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "All_Beauty": {
                        "num_train_tokens":10_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Appliances": {
                        "num_train_tokens":18_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "AMAZON_FASHION": {
                        "num_train_tokens":22_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Digital_Music": {
                        "num_train_tokens":55_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Arts_Crafts_and_Sewing": {
                        "num_train_tokens":80_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Industrial_and_Scientific": {
                        "num_train_tokens":55_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Luxury_Beauty": {
                        "num_train_tokens":20_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "code_contests": {
                        "num_train_tokens":  1_900_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000},
                "Magazine_Subscriptions": {
                        "num_train_tokens":1_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Automotive": {
                        "num_train_tokens":245_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "CDs_and_Vinyl": {
                        "num_train_tokens":400_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Musical_Instruments": {
                        "num_train_tokens":70_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Cell_Phones_and_Accessories": {
                        "num_train_tokens":380_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Prime_Pantry": {
                        "num_train_tokens":8_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Kindle_Store": {
                        "num_train_tokens":400_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Software": {
                        "num_train_tokens":30_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Patio_Lawn_and_Garden": {
                        "num_train_tokens":200_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Office_Products": {
                        "num_train_tokens":200_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Pet_Supplies": {
                        "num_train_tokens":280_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Movies_and_TV": {
                        "num_train_tokens":495_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Video_Games": {
                        "num_train_tokens":185_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Tools_and_Home_Improvement": {
                        "num_train_tokens":350_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Toys_and_Games": {
                        "num_train_tokens":290_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Home_and_Kitchen": {
                        "num_train_tokens":820_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Sports_and_Outdoors": {
                        "num_train_tokens":500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Electronics": {
                        "num_train_tokens":1_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Clothing_Shoes_and_Jewelry": {
                        "num_train_tokens":950_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Books": {
                        "num_train_tokens":4_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "Books": {
                        "num_train_tokens":4_000_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                 "stackexchange":{
                        "num_train_tokens":1_500_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                 "ala_text_20200604":{
                        "num_train_tokens":170_000_000,
                        "num_dev_tokens": 1_000_000,
                        "num_test_tokens": 1_000_000
                },
                "haw_text_20200604":{
                                "num_train_tokens":30_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "minn_text_20200604":{
                                "num_train_tokens":95_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "okla_text_20200604":{
                                "num_train_tokens":120_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "tenn_text_20200604":{
                                "num_train_tokens":85_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "alaska_text_20200604":{
                                "num_train_tokens":30_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "idaho_text_20200604":{
                                "num_train_tokens":45_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "miss_text_20200604":{
                                "num_train_tokens":105_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "or_text_20200604":{
                                "num_train_tokens":110_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "tex_text_20200604":{
                                "num_train_tokens":495_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "am-samoa_text_20200604":{
                                "num_train_tokens":1_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ind_text_20200604":{
                                "num_train_tokens":150_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "mo_text_20200604":{
                                "num_train_tokens":265_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "pa_text_20200604":{
                                "num_train_tokens":340_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },

                "ariz_text_20200604":{
                                "num_train_tokens":60_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "iowa_text_20200604":{
                                "num_train_tokens":100_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "mont_text_20200604":{
                                "num_train_tokens":60_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "pr_text_20200604":{
                                "num_train_tokens":80_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },

                "cal_text_20200604":{
                                "num_train_tokens":400_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "kan_text_20200604":{
                                "num_train_tokens":95_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "n-mar-i_text_20200604":{
                                "num_train_tokens":4_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ri_text_20200604":{
                                "num_train_tokens":30_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "us_text_20200604":{
                                "num_train_tokens":4_000_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "colo_text_20200604":{
                                "num_train_tokens":80_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ky_text_20200604":{
                                "num_train_tokens":115_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "sc_text_20200604":{
                                "num_train_tokens":65_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "utah_text_20200604":{
                                "num_train_tokens":55_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "conn_text_20200604":{
                                "num_train_tokens":100_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "la_text_20200604":{
                                "num_train_tokens":360_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "nd_text_20200604":{
                                "num_train_tokens":40_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "sd_text_20200604":{
                                "num_train_tokens":34_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "va_text_20200604":{
                                "num_train_tokens":80_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "neb_text_20200604":{
                                "num_train_tokens":70_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "vi_text_20200604":{
                                "num_train_tokens":5_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "dc_text_20200604":{
                                "num_train_tokens":100_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "mass_text_20200604":{
                                "num_train_tokens":140_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "nev_text_20200604":{
                                "num_train_tokens":20_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "vt_text_20200604":{
                                "num_train_tokens":25_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "del_text_20200604":{
                                "num_train_tokens":30_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "md_text_20200604":{
                                "num_train_tokens":125_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "nh_text_20200604":{
                                "num_train_tokens":25_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "w-va_text_20200604":{
                                "num_train_tokens":65_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "fla_text_20200604":{
                                "num_train_tokens":220_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "me_text_20200604":{
                                "num_train_tokens":40_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "nj_text_20200604":{
                                "num_train_tokens":170_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "wash_text_20200604":{
                                "num_train_tokens":140_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ga_text_20200604":{
                                "num_train_tokens":190_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ny_text_20200604":{
                                "num_train_tokens":695_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "wis_text_20200604":{
                                "num_train_tokens":100_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "mich_text_20200604":{
                                "num_train_tokens":120_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "ohio_text_20200604":{
                                "num_train_tokens":180_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
                "wyo_text_20200604":{
                                "num_train_tokens":30_000_000,
                                "num_dev_tokens": 1_000_000,
                                "num_test_tokens": 1_000_000
                        },
        "2007scape": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "airsoft": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "anime": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "apexlegends": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "apple": {"num_train_tokens": 2_500_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "argentina": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "army": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "askgaybros": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "asoiaf": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "asoiafcirclejerk": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "assholedesign": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "atheism": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "australia": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "aww": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "bangtan": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "barstoolsports": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "baseball": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "bjj": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "blogsnark": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "boardgames": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "books": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "bostonceltics": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "boxoffice": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "brasil": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "btc": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "buildapc": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "canada": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cars": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cats": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "changemyview": {"num_train_tokens": 11_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "childfree": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "churning": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "classicwow": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "comedyheaven": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "confession": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "confessions": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "conspiracy": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "copypasta": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cscareerquestions": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cursedcomments": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cursedimages": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dankmemes": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dataisbeautiful": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dating_advice": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "datingoverthirty": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dauntless": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "de": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "deadbydaylight": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "depression": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dndnext": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "elderscrollsonline": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "entitledparents": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "europe": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "exmormon": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "explainlikeimfive": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "facepalm": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "fantasybaseball": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "fatlogic": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ffxiv": {"num_train_tokens": 14_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "financialindependence": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "fireemblem": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "fo76": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "forhonor": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "formula1": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "france": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "freefolk": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "funny": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gameofthrones": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gaming": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gardening": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gatekeeping": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gifs": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "golf": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gonewild": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "grandorder": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gtaonline": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "guns": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "hearthstone": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "hiphopheads": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "hockey": {"num_train_tokens": 13_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "iamatotalpieceofshit": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "india": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "insanepeoplefacebook": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "interestingasfuck": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "investing": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ireland": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "italy": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "jailbreak": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "keto": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "kpop": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "lakers": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "leagueoflegends": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "legaladvice": {"num_train_tokens": 13_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "loseit": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "manga": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "marvelstudios": {"num_train_tokens": 14_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "me_irl": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "memes": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "mildlyinfuriating": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "mildlyinteresting": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "motorcycles": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "movies": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "nba": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "neoliberal": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "news": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "newzealand": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "nfl": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "niceguys": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "nottheonion": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "oculus": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "oddlysatisfying": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "offmychest": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pathofexile": {"num_train_tokens": 14_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pcgaming": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pcmasterrace": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "personalfinance": {"num_train_tokens": 16_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pics": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pokemon": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "politics": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "popheads": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "quityourbullshit": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "raisedbynarcissists": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "reddeadredemption": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "reddevils": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "relationship_advice": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "relationships": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "runescape": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "rupaulsdragrace": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "samharris": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "science": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "sex": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "singapore": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "smashbros": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "soccer": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "space": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "starcitizen": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "starterpacks": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "stopdrinking": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "survivor": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "sysadmin": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "technology": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "techsupport": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "teenagers": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "television": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "teslamotors": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "thebachelor": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "thedivision": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "therewasanattempt": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "tifu": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "todayilearned": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "toronto": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "torontoraptors": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "totalwar": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "traaaaaaannnnnnnnnns": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "trashy": {"num_train_tokens": 10_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "trees": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "tumblr": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ukpolitics": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "unitedkingdom": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "unpopularopinion": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "vancouver": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "vegan": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "videos": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "wallstreetbets": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "weddingplanning": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "whatisthisthing": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "wholesomememes": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "whowouldwin": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "worldnews": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "worldpolitics": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "wow": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "xboxone": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "yugioh": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "academia.stackexchange": {"num_train_tokens": 12_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "android.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "apple.stackexchange": {"num_train_tokens": 12_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "arduino.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "askubuntu": {"num_train_tokens":  42_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "astronomy.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "aviation.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "bicycles.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "biology.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "bitcoin.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "blender.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "boardgames.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "chemistry.stackexchange": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "christianity.stackexchange": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "codegolf.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "codereview.stackexchange": {"num_train_tokens": 47_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cooking.stackexchange": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "crypto.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cs.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "cstheory.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "datascience.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dba.stackexchange": {"num_train_tokens": 20_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "diy.stackexchange": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "drupal.stackexchange": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "dsp.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "electronics.stackexchange": {"num_train_tokens": 35_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ell.stackexchange": {"num_train_tokens": 11_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "emacs.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "english.stackexchange": {"num_train_tokens": 20_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "es.stackoverflow": {"num_train_tokens":  25_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "stackoverflow": {"num_train_tokens":  3_300_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ethereum.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "french.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gamedev.stackexchange": {"num_train_tokens": 11_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gaming.stackexchange": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gardening.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "german.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "gis.stackexchange": {"num_train_tokens": 17_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "graphicdesign.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "hermeneutics.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "hinduism.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "history.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "interpersonal.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "japanese.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "judaism.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "law.stackexchange": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "magento.stackexchange": {"num_train_tokens": 8_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "mathematica.stackexchange": {"num_train_tokens": 18_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "mathoverflow": {"num_train_tokens":  32_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "mechanics.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "money.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "movies.stackexchange": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "music.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "networkengineering.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "parenting.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "philosophy.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "photo.stackexchange": {"num_train_tokens": 6_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "physics.stackexchange": {"num_train_tokens": 41_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "politics.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "pt.stackoverflow": {"num_train_tokens": 28_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "puzzling.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "quant.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "raspberrypi.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "rpg.stackexchange": {"num_train_tokens": 23_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ru.stackoverflow": {"num_train_tokens":  44_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "salesforce.stackexchange": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "scifi.stackexchange": {"num_train_tokens": 23_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "security.stackexchange": {"num_train_tokens": 15_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "serverfault": {"num_train_tokens":  49_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "sharepoint.stackexchange": {"num_train_tokens": 7_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "skeptics.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "softwareengineering.stackexchange": {"num_train_tokens": 22_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "space.stackexchange": {"num_train_tokens": 4_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "stats.stackexchange": {"num_train_tokens": 31_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "superuser": {"num_train_tokens": 57_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "tex.stackexchange": {"num_train_tokens": 55_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "travel.stackexchange": {"num_train_tokens": 9_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "unix.stackexchange": {"num_train_tokens": 36_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "ux.stackexchange": {"num_train_tokens": 5_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "webapps.stackexchange": {"num_train_tokens": 1_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "webmasters.stackexchange": {"num_train_tokens": 2_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "wordpress.stackexchange": {"num_train_tokens": 14_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "workplace.stackexchange": {"num_train_tokens": 14_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "worldbuilding.stackexchange": {"num_train_tokens": 21_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
        "writers.stackexchange": {"num_train_tokens": 3_000_000, "num_dev_tokens": 1_000_000, "num_test_tokens": 1_000_000},
}
