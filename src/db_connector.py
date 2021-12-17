import psycopg2 as psg
import pandas as pd
import passes


def process_words_table(connection):
    words = pd.read_csv('../data/words.txt', header=None, delimiter=",")
    cursor = connection.cursor()
    for col in words:
        cursor.execute("INSERT INTO words (id, word) VALUES(%s, %s)", (col + 1, str(words[col].values[0])))
    cursor.close()


def process_metadata_test(connection):
    metadata_train = pd.read_csv('../data/example_result_test.txt', delimiter=",")
    cursor = connection.cursor()

    for index, data in metadata_train.iterrows():
        genre_cd1_1 = data["genre_cd1_1"] if pd.notnull(data["genre_cd1_1"]) else ''
        genre_cd1_2 = data["genre_cd1_2"] if pd.notnull(data["genre_cd1_2"]) else ''
        genre_cd2_1 = data["genre_cd2_1"] if pd.notnull(data["genre_cd2_1"]) else ''
        genre_cd2_2 = data["genre_cd2_2"] if pd.notnull(data["genre_cd2_2"]) else ''
        cursor.execute(
            "INSERT INTO song_genre_metadata (song_id_mxm, song_id_msd, genre_cd1_1, genre_cd1_2, genre_cd2_1, genre_cd2_2, istrain) VALUES(%s, %s, %s, %s, %s, %s, %s)",
            (data[0], '', genre_cd1_1, genre_cd1_2, genre_cd2_1, genre_cd2_2, "False"))


def process_metadata(connection):
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    cursor = connection.cursor()

    for index, data in metadata_train.iterrows():
        genre_cd1_1 = data["genre_cd1_1"] if pd.notnull(data["genre_cd1_1"]) else ''
        genre_cd1_2 = data["genre_cd1_2"] if pd.notnull(data["genre_cd1_2"]) else ''
        genre_cd2_1 = data["genre_cd2_1"] if pd.notnull(data["genre_cd2_1"]) else ''
        genre_cd2_2 = data["genre_cd2_2"] if pd.notnull(data["genre_cd2_2"]) else ''
        cursor.execute(
            "INSERT INTO song_genre_metadata (song_id_mxm, song_id_msd, genre_cd1_1, genre_cd1_2, genre_cd2_1, genre_cd2_2, istrain) VALUES(%s, %s, %s, %s, %s, %s, %s)",
            (data[0], '', genre_cd1_1, genre_cd1_2, genre_cd2_1, genre_cd2_2, "True"))


def process_word_test(connection):
    metadata_test = pd.read_csv('../data/example_result_test.txt', delimiter=",")
    cursor = connection.cursor()

    for index, data in metadata_test.iterrows():
        print(f"#####{data[0]}")
        for i in range(6, data.size):
            if not pd.notnull(data[i]):
                break
            id = data[i].split(":")[0]
            count = data[i].split(":")[1]
            print(f"id: {id} count: {count}")
            cursor.execute(
                "INSERT INTO song_words (song_id_mxm, word_id, count) VALUES(%s, %s, %s)",
                (data[0], id, count))


def process_word_train(connection):
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    cursor = connection.cursor()

    for index, data in metadata_train.iterrows():
        for i in range(6, data.size):
            if not pd.notnull(data[i]):
                break
            id = data[i].split(":")[0]
            count = data[i].split(":")[1]
            cursor.execute(
                "INSERT INTO song_words (song_id_mxm, word_id, count) VALUES(%s, %s, %s)",
                (data[0], id, count))




if __name__ == '__main__':
    conn = psg.connect(
        host=passes.db_host,
        database=passes.db_database,
        user=passes.db_username,
        password=passes.db_pass)
