def load_movie_list():
    """
    Load movie list from movie_ids.txt.

    Returns
    -------
    movie_list : list
        The movie list.
    """
    movie_list =[]
    with open("movie_ids.txt") as f:
        for line in f:
            movie_list.append(line[line.index(' ') + 1:].rstrip())
    return movie_list
