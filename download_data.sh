wget https://nlpcrossworddata.blob.core.windows.net/test2/data-20200318T221609Z-001.zip
unzip data-20200318T221609Z-001.zip
cp guardian_data/test_guardian_600.tok data/full_data

cd data/embeddings
wget https://nlpcrossworddata.blob.core.windows.net/test2/glove.6B.zip
unzip glove.6B.zip

cd ../../subword-nmt
git clone https://github.com/rsennrich/subword-nmt.git

cd ..
