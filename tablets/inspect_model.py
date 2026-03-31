import joblib

# Load the files
le_brand   = joblib.load('final_project/tablets/le_brand.pkl')
le_website = joblib.load('final_project/tablets/le_website.pkl')
model      = joblib.load('final_project/tablets/tablet_model.pkl')

# Inspect them
print("Brands:", le_brand.classes_)
print("Websites:", le_website.classes_)
print("Model type:", type(model))
print("Model params:", model.get_params())