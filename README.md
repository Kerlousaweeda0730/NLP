Kerlous Aweeda - NLP

# Instructions:

<p> Required files: (in the same directory) </p>
<ol>
<li>NB.py</li>
<li>pre-process.py</li>
<li>movie-review-HW2 folder (as downloaded)</li>
<li>small-corpus-reviews folder (included in tar)</li>
</ol>
<p>Generated by pre-process.py (also included in tar)</p>
<ol>
<li>In both movie-review-HW2 and small-corpus-reviews, a feature_vectors folder is created</li>
<li>Both have a test and train folder</li>
<li>In the test and train folder are the vector representations of each movie review</li>
<li>test_output_vector.json (generated & included)</li>
<li>train_output_vector.json (generated & included)</li>
</ol>
<p>Generated by NB.py (also included in tar)</p>
<ol>
<li>In both movie-review-HW2 and small-corpus-reviews, a movie-rewview-BOW.NB file is created containing the log-probabilities of all the parameters.</li>
<li>Both folders also include an output.txt file revealing predictions, results, and the accuracy at the bottom.</li>
</ol>

## _You should run the pre-process.py file first, then run the NB.py file._

<p>However, going out of order should not cause errors due to the inclusion of all generated files in the tarball</p>

## If you encounter an issue with compiling

<p>I've run into an issue with python versioning where the line:</p>
<code>with open ('XYZ.txt','r', encoding='utf8') as f:</code>
<p>will fail to compile due to the encoding parameter not being required, however, for my machine it was required. I'm using Python on a windows machine.
I hope not but if for any reason this causes an issue, please use the find and replace feature to replace:
<code>, encoding='utf8'</code>
with a space to remove it.</p>
