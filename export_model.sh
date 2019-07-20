#!/bin/sh

set -e

mkdir -p models
if [ $# -lt 1 ]; then
    echo "Export model using default sequence length 384"
    python sentence_embedding/bert/export/export.py --task classification --output_dir models/384
    python sentence_embedding/bert/export/export.py --task regression --output_dir models/384
    python sentence_embedding/bert/export/export.py --task question_answering --output_dir models/384
else
    echo "Export model using seq_length $1"
    python sentence_embedding/bert/export/export.py --task classification --seq_length $1 --output_dir models/$1
    python sentence_embedding/bert/export/export.py --task regression --seq_length $1 --output_dir models/$1
    python sentence_embedding/bert/export/export.py --task question_answering --seq_length $1 --output_dir models/$1
fi
