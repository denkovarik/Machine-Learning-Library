
# Move to testing directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

echo "Running Tests for Perceptron Module"
python3 perceptronTests.py
echo ""
echo "Running Tests for K Nearest Neighbors Module"
python3 testKNN.py 
