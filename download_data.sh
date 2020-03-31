wget https://nlpcrossworddata.blob.core.windows.net/test2/data-20200318T221609Z-001.zip -q
unzip data-20200318T221609Z-001.zip
cp guardian_data/test_guardian_600.tok data/full_data

wget https://raw.githubusercontent.com/ucabops/robbie/master/data/gquick-10-entries.txt -P data/full_data -q
wget https://nlpcrossworddata.blob.core.windows.net/test2/definitions_100000.vocab -P data/full_data -q
wget https://nlpcrossworddata.blob.core.windows.net/test2/top_candidate_results.txt -P data/full_data -q
wget https://nlpcrossworddata.blob.core.windows.net/test2/final_csvsx_word.csv -q

cd data/embeddings
wget https://nlpcrossworddata.blob.core.windows.net/test2/glove.6B.zip -q
unzip glove.6B.zip

cd ../../subword-nmt
git clone https://github.com/rsennrich/subword-nmt.git

cd ../models

wget https://nlpcrossworddata.blob.core.windows.net/test2/avg_word.zip -q
unzip avg_word.zip

cd ..
