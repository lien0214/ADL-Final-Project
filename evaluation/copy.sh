rm -rf predictions
mkdir -p predictions
cp -r ../mt5/predictions/* predictions/
cp -r ../llama/prediction/* predictions/
files=$(ls predictions)
for file in $files
do
    if [[ $file == *_* ]]; then
        data_type="${file%%_*}"
        batch="${file#*_}"
        mv predictions/$file "predictions/llama-$data_type-$batch"
    else
        data_type="${file%%-*}"
        batch="${file#*-}"
        mv predictions/$file "predictions/mt5-$data_type-$batch"
    fi
done