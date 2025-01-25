RUN_NAME=$1

LLAMA_CPP_DIR=/home/ubuntu/llama.cpp

set -e

if [ -z "$RUN_NAME" ]
then
    echo "Please provide the run name"
    exit 1
fi

OUT_FILE_NAME=$RUN_NAME/compiled.gguf

if [ ! -e $OUT_FILE_NAME ]
then
    echo "Compiling GGUF, saving to $OUT_FILE_NAME"
    python $LLAMA_CPP_DIR/convert_hf_to_gguf.py $RUN_NAME --outfile $OUT_FILE_NAME
fi

echo "Starting server"
source .venv/bin/activate
python3 -m llama_cpp.server --model $OUT_FILE_NAME --hf_tokenizer_config_path $RUN_NAME/tokenizer_config.json