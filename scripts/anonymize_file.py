import re
import argparse
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()
    re1 = {
            "regex": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
            "repl": "<|EMAIL|>"
        }

    re2 = {"regex": "[0-9]{10}-[0-9A-Fa-f]{53}", "repl":"<|DART|>"}

    re3 = {"regex": "@\[[0-9]+:[0-9]+:(?=[^\]])(([^\\:\]]*(?:\\.)*)*)\]", "repl": "<|FBUSERID|>"}

    re4 ={"regex": "(?:(?<![\d-])(?:\+\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-\/.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))", "repl": "<|PHONE_NUMBER|>"}

    re5 ={"regex": "(?:4\d{12}(?:\d{3})?|(?:5[1-5]\d{2}|222[1-9]|22[3-9]\d|2[3-6]\d{2}|27[01]\d|2720)\d{12}|3[47]\d{13}|5019\d{12}|3(?:0[0-5]|[68]\d)\d{11}|6(?:011|5\d{2})\d{12}|(?:2131|1800|35\d{3})\d{11})", "repl": "<CREDIT_CARD_NUMBER>"}

    re6 ={"regex": "(?!(?:000|666|9))\d{3}-(?!00)\d{2}-(?!0000)\d{4}", "repl": "<|SSN|>"}

    re7 = {"regex": "\d+\s(?:(?:[a-z0-9.-]+[ ]?)+\s(?:Alley|Aly|Ave(?:nue)?|Boulevard|Blvd|Br(?:anch)?|Center|Ctr|Cir(?:cle)?|Court|Ct|Crossing|Xing|Dr(?:ive)?|Est(?:ate)?|Expressway|Expy|Freeway|Fwy|Highway|Hwy|Hills|Hls|Knoll|Knl|Landing|Lndg|Lane|Ln|Manor|Mnr|Meadow|Mdw|Parkway|Pkwy|Pass|Path|Plaza|Plz|Road|Rd|Run|Sq(?:uare)?|St(?:ation|reet|a)?|Ter(?:ace)?|Trail|Trl|Turnpike|Tpke|Valley|Vly|View|Vw|Village|Vlg|Vis(?:ta)?|Walk|Way)|(?:Route|Rte|Interstate|I)[- ]?\d{1,3})(?:\s(?:Apt[\.]?|Apartment|#)[ ]?\d+[a-z]?)?(?:\s(?:[a-z-]+[ ]?)+,?(?:\s(?:AK|AL(?:aska|abama)?|AR(?:kansas|izona)?|AZ|CA(?:lifornia)?|CO(?:lorado|nnecticut)?|CT|DC|DE(?:laware)?|FL(?:orida)?|GA|Georgia|GU(?:am)?|HI|Hawaii|IA|Iowa|ID(?:aho)?|IL(?:linois)?|IN(?:diana)?|KS|Kansas|KY|Kentucky|LA|Louisiana|MA(?:ssachusetts|ryland|ine)?|MD|ME|MI(?:chigan|nnesota|ssissippi|ssouri)|MN|MO(?:ntana)?|MS|MT|NC|North[ ]Carolina|ND|North[ ]Dakota|NH|New[ ]Hampshire|NJ|New[ ]Jersey|NM|New[ ]Mexico|NV|Nevada|NY|New[ ]York|OH(?:io)?|OK(?:lahoma)?|OR(?:egon)?|PA|Pennsylvania|PR|Puerto[ ]Rico|RI|Rhode[ ]Island|SC|South[ ]Carolina|SD|South[ ]Dakota|TN|Tennessee|TX|Texas|UT(?:ah)?|VA|Virginia|VI(?:rgin[ ]Islands)?|VT|Vermont|WA(?:shington(?:[ ]D[. ]?C[.]?)?)?|WI(?:sconsin)?|WV|West[ ]Virginia|WY(?:oming)?)(?:\s\b\d{5}(?:-\d{4})?\b)?)?)?", 
        "repl": "<|ADDRESS|>"}

    re8 = {"regex": "@[a-zA-Z0-9_\.\-]{1,30}", "repl": "@USER"}


    re_list = [re1,re2,re3,re4,re5,re6,re8]


    anonymizer = {re.compile(x['regex']): x['repl'] for x in re_list}

    with open(args.input_file, 'r') as f, open(args.output_file, 'w+') as g:
        for line in tqdm(f.readlines()):
            for x,y in anonymizer.items():
                line = x.sub(y, line)
            g.write(line)