echo "Which Approach you wanna use :\n(1)Naive bayesian\n(2)Decision Tree\n :"
read -p "Enter choice :" choice
if [ $choice = 1 ]
then
   echo "Naive bayes file loading..."
   python src/naiveBayes.py
else 
   echo "Decision Tree file loading..."
   python src/decisionTree.py
fi       
$SHELL

