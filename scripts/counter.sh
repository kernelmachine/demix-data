cd /private/home/suching/raw_data/s2orc
mkdir Art/Art 
mv Art/*.gz Art/Art/
mkdir Biology/Biology
mv Biology/*.gz Biology/Biology
mkdir Business/Business 
mv Business/*.gz Business/Business 
mkdir Chemistry/Chemistry 
mv Chemistry/*.gz Chemistry/Chemistry 
mkdir cs/cs 
mv cs/*.gz cs/cs 
mkdir Economics/Economics 
mv Economics/*.gz Economics/Economics 
mkdir Engineering/Engineering 
mv Engineering/*.gz Engineering/Engineering 
mkdir env_sci/env_sci 
mv env_sci/*.gz env_sci/env_sci 
mkdir Geography/Geography 
mv Geography/*.gz Geography/Geography 
mkdir Geology/Geology 
mv Geology/*.gz Geology/Geology 
mkdir History/History 
mv History/*.gz History/History 
mkdir materials/materials 
mv materials/*.gz materials/materials 
mkdir Mathematics/Mathematics 
mv Mathematics/*.gz Mathematics/Mathematics 
mkdir Medicine/Medicine 
mv Medicine/*.gz Medicine/Medicine 
mkdir Philosophy/Philosophy 
mv Philosophy/*.gz Philosophy/Philosophy 
mkdir Physics/Physics 
mv Physics/*.gz Physics/Physics 
mkdir polisci/polisci 
mv polisci/*.gz polisci/polisci 
mkdir Psychology/Psychology 
mv Psychology/*.gz Psychology/Psychology 
mkdir Sociology/Sociology
mv Sociology/*.gz Sociology/Sociology

cd /private/home/suching/demix-data/

python -m domain_loader.count_words --domain Art --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Biology --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Business --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Chemistry --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain cs --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Economics --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Engineering --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain env_sci --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Geography --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Geology --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain History --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain materials --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Mathematics --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Medicine --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Philosophy --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Physics --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain polisci --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Psychology --batch-size 1024 --num-workers 2 --use-iterable-dataset
python -m domain_loader.count_words --domain Sociology --batch-size 1024 --num-workers 2 --use-iterable-dataset
