"""Everything the pack remembers between prompts.

:mod:`.store` is the sqlite database and :mod:`.migration` fills it once from the JSON
files. :mod:`.database` is the key and value surface, :mod:`.tokens` holds ``[time]``
substitution and :mod:`.history` the recent-path lists.
"""
