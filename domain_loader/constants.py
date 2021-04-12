from pathlib import Path
PROJECT_DIR = Path("/private/home/suching/raw_data/")


TOKEN_COUNTS = {"1b": {'num_train_tokens': 700_000_000, 
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
                }
                
