# from flask import Flask, request, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# @app.route('/recommend', methods=['POST'])
# def recommend_alumni():

#     data = request.json

#     print(data)

#     student_interests = " ".join(data['studentInterests'])

#     alumni_list = data['alumni']

#     alumni_skills = [
#         " ".join(alumni['skills'])
#         for alumni in alumni_list
#     ]

#     documents = [student_interests] + alumni_skills

#     vectorizer = TfidfVectorizer()

#     tfidf_matrix = vectorizer.fit_transform(documents)

#     similarity_scores = cosine_similarity(
#         tfidf_matrix[0:1],
#         tfidf_matrix[1:]
#     )[0]

#     recommendations = []

#     for index, score in enumerate(similarity_scores):

#         if score > 0:

#             recommendations.append({
#                 "alumniId": alumni_list[index]['id'],
#                 "score": round(float(score), 2)
#             })

#     recommendations.sort(
#         key=lambda x: x['score'],
#         reverse=True
#     )

#     recommendations = recommendations[:5]

#     return jsonify(recommendations)

# if __name__ == '__main__':
#     app.run(debug=True, port=7000)

from flask import Flask, request, jsonify

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# LOAD MODEL
model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

embedding_cache = {}

@app.route('/recommend', methods=['POST'])
def recommend_alumni():
    data = request.json

    # STUDENT INTERESTS
    student_interests = " ".join([
        str(item)
        for item in data['studentInterests']
        if item
    ])

    alumni_list = data['alumni']

    # ALUMNI SKILLS
    alumni_skills = [
        " ".join([
            str(skill)
            for skill in alumni['skills']
            if skill
        ])
        for alumni in alumni_list
    ]

    # CREATE EMBEDDINGS
    student_embedding = model.encode(
        [student_interests]
    )

    alumni_embeddings = model.encode(
        alumni_skills
    )

    # CALCULATE SIMILARITY
    similarity_scores = cosine_similarity(
        student_embedding,
        alumni_embeddings
    )[0]

    recommendations = []

    for index, score in enumerate(similarity_scores):

        if score > 0.2:
            matched_score = calculate_skill_matched(data['studentInterests'], alumni_list[index]['skills'])

            recommendations.append({

                "alumniId": alumni_list[index]['id'],

                "semanticScore": round(
                    float(score),
                    2
                ),

                "matchPercentage": matched_score
            })

    recommendations.sort(
        key=lambda x: x['semanticScore'],
        reverse=True
    )

    recommendations = recommendations[:5]

    return jsonify(recommendations)

def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text)
    return embedding_cache[text]

def calculate_skill_matched(student_skills, alumni_skills): # Brute force
    matched = 0
    for student_skill in student_skills:
        student_embedding = get_embedding(student_skill)
        is_matched = False
        for alumni_skill in alumni_skills:
            alumni_embedding = get_embedding(alumni_skill)
            similarity = cosine_similarity([student_embedding], [alumni_embedding])[0][0]
            if similarity > 0.5:
                is_matched = True
                break
        if is_matched:
            matched += 1
    if len(student_skills) == 0:
        return 0
    return round((matched / len(student_skills)) * 100)


if __name__ == '__main__':
    app.run(
        debug=True,
        port=7000
    )