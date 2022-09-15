NAME=$(head -n 1 environment.yaml | cut -f 2 -d ' ')
echo "Running $NAME"
echo "/opt/conda/envs/$(head -n 1 environment.yaml | cut -f 2 -d ' ')/bin/$@"