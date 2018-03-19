# Music_Recommender_PySpark
Used PySpark/MLlib to build a collaborative filtering based music recommender system

## Data

Audioscrobbler Dataset, available here : http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html

user_artist_data.txt contains a user ID, an artist ID, and a play count.

artist_data.txt contains the IDs associated with artist names.

artist_alias.txt : contains artist IDs that may be misspelled or nonstandard to the ID of the artist’s canonical name. It contains two IDs per line, separated by a tab.

## Technology Used
- Python 2.7
- PySpark (MLlib)