# objectify/1 takes an array of string values as inputs, converts
# numeric values to numbers, and packages the results into an object
# with keys specified by the "headers" array
def objectify(headers):
  # For jq 1.4, replace the following line by: def tonumberq: .;
  def tonumberq: tonumber? // .;
  . as $in
  | reduce range(0; headers|length) as $i ({}; .[headers[$i]] = ($in[$i] | tonumberq) );

def csv2table:
  # For jq 1.4, replace the following line by:  def trim: .;
  def trim: sub("^ +";"") |  sub(" +$";"");
  split("\n") | map( split(",") | map(trim) );

def csv2json:
  csv2table
  | .[0] as $headers
  | reduce (.[1:][] | select(length > 0) ) as $row
      ( []; . + [ $row|objectify($headers) ]);

csv2json
