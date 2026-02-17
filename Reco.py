import pandas as pd

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\archive\udemy_courses.csv")

df = df[['course_title', 'url', 'num_subscribers', 'subject', 'level', 'is_paid']]

def chatbot():
    print("🤖 Welcome to the Course Recommendation Chatbot!\n")
    
    print("Available Subjects:")
    print(df['subject'].unique())
    
    subject = input("\nEnter your preferred subject: ").strip()
    level = input("Enter your level (Beginner Level / Intermediate Level / All Levels): ").strip()
    paid_input = input("Do you want a paid course? (yes/no): ").strip().lower()
    
    is_paid = True if paid_input == "yes" else False
    
    results = df[
        (df['subject'].str.lower() == subject.lower()) &
        (df['level'].str.lower() == level.lower()) &
        (df['is_paid'] == is_paid)
    ].sort_values(by='num_subscribers', ascending=False)
    
    if results.empty:
        print("\n❌ No courses found. Try different options.")
    else:
        print("\n🎓 Recommended Courses:\n")
        for _, row in results.head(5).iterrows():
            print(f"📘 Course: {row['course_title']}")
            print(f"🔗 Link: {row['url']}")
            print(f"👥 Subscribers: {row['num_subscribers']}")
            print("-" * 40)

# Run chatbot
chatbot()
