import re
import mysql.connector
import markdown
import feedparser
import json
import os
import random
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import os
from datetime import datetime
import requests
def insert_blog_post_to_db(title, summary, content, keywords, slug, thumbnail):
    # Fetch MySQL credentials from environment variables
    mysql_host = os.getenv('MYSQL_HOST')
    mysql_user = os.getenv('MYSQL_USER')
    mysql_password = os.getenv('MYSQL_PASSWORD')
    mysql_database = os.getenv('MYSQL_DATABASE')

    # Connect to the remote MySQL database
    db = mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database
    )

    cursor = db.cursor()
    created_at = updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    page_sql = """
    INSERT INTO `pages` (
        `id`, `slug`, `target`, `type`, `featured_image`, 
        `tool_name`, `icon_image`, `custom_tool_link`, `post_status`, 
        `page_status`, `tool_status`, `ads_status`, `popular`, 
        `position`, `category_id`, `created_at`, `updated_at`
    ) 
    VALUES (
        NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    page_values = (
        re.sub(r'[^a-zA-Z0-9\s-]', '', slug.replace('The title is', '').replace('The title of this blog post is', '')).lower().strip().replace('\n', ' ').replace(' ', '-').replace('the-title-of-this-polished-and-professional-blog-post-is', ""), "_self", "post", "https://multiculturaltoolbox.com/assets/img/nastuh.jpg",
        None, None, None,
        1, 1,1,
        1, 1, 1,
        None, created_at, updated_at
    )

    # Execute the page insertion
    cursor.execute(page_sql, page_values)
    db.commit()

    # Get the last inserted ID for `pages`
    page_id = cursor.lastrowid
    print(f"Inserted page ID: {page_id}")

    # Insert current timestamp for created_at and updated_at
    

    # SQL query to insert the generated blog post

    sql = """
    INSERT INTO `page_translations` (
        `locale`, 
        `page_title`, 
        `robots_meta`, 
        `sitename_status`, 
        `site_name_status`, 
        `title`, 
        `subtitle`, 
        `short_description`, 
        `description`, 
        `page_id`, 
        `created_at`, 
        `updated_at`
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        "en", title.replace('The title is', '').replace('I think your revised blog post looks great', '').replace("Here is a rewritten version of the blog post with a polished and professional tone, grammar, and readability", '').replace('The title you provided is', '').replace('Here is the polished and professional version of the blog post', '').replace("Here's the revised blog post", '').replace('The title of this blog post is', '').strip('"').replace('\n', ' '), 1, 1,
        1, title.replace('The title is', '').replace('I think your revised blog post looks great', '').replace("Here is a rewritten version of the blog post with a polished and professional tone, grammar, and readability", '').replace('The title you provided is', '').replace('Here is the polished and professional version of the blog post', '').replace("Here's the revised blog post", '').strip('"').replace('\n', ' '), title.replace('The title is', '').replace('I think your revised blog post looks great', '').replace("Here is a rewritten version of the blog post with a polished and professional tone, grammar, and readability", '').replace('The title you provided is', '').replace('Here is the polished and professional version of the blog post', '').replace("Here's the revised blog post", '').strip('"').replace('\n', ' '), title.replace('The title is', '').replace('I think your revised blog post looks great', '').replace("Here is a rewritten version of the blog post with a polished and professional tone, grammar, and readability", '').replace('The title you provided is', '').replace('Here is the polished and professional version of the blog post', '').replace("Here's the revised blog post", '').strip('"').replace('\n', ' '),
        markdown.markdown(content).replace('Here is the edited blog post', '').replace("Here's is the edited blog post", '').replace("Here is a rewritten version of the blog post with a polished and professional tone, grammar, and readability", '').replace("Here's the revised blog post", '').replace('Here is the revised blog post', '').replace('\n', '<br>').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', ''), page_id, created_at, updated_at
    )
  
    cursor.execute(sql, values)
    db.commit()

    print(f"Blog post '{title}' inserted successfully!")

    # Close the cursor and connection
    cursor.close()
    db.close()
    print(f"successfully!")
    API_KEY = "7ab877d3e5d04c1a8952e29d9ce9e7e6"
    INDEXNOW_ENDPOINTX = f"https://multiculturaltoolbox.com/blog/{pgSlug}"


# URL to be indexed
    URLS = [INDEXNOW_ENDPOINTX]

    # IndexNow API Endpoint
    INDEXNOW_ENDPOINT = f"https://api.indexnow.org/indexnow"

    # Prepare data payload
    payload = {
        "host": "multiculturaltoolbox.com",  # Replace with your actual domain
        "key": API_KEY,
        "keyLocation": f"https://multiculturaltoolbox.com/{API_KEY}.txt",  # Place API key on your site
        "urlList": URLS
    }
    
    
def download_image(image_url, file_name):
    response = requests.get(image_url)
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print(f'Download Completed: {file_name}')
width = 720
height = 1280
model = 'flux'
# Read seed from file or initialize it
seed_file = 'seed.txt'
if os.path.exists(seed_file):
    with open(seed_file, 'r') as file:
        seed = int(file.read().strip()) + 1
else:
    seed = 1
rss_url = 'https://www.manilatimes.net/world/feed/'
feed = feedparser.parse(rss_url)

# Parse the RSS feed entries
new_entries = []
new_guids = set()

for entry in feed.entries:
    new_entries.append({
        'title': entry.title,
        'link': entry.link,
        'description': entry.description,
        'published': entry.published,
        'guid': entry.guid  # Unique identifier for each item
    })
    new_guids.add(entry.guid)

# File paths
file_path = 'rss_data.json'
guids_file_path = 'guids.json'

# If the file exists, read the existing data
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        existing_entries = json.load(f)
else:
    existing_entries = []

# If the GUIDs file exists, read the existing GUIDs
if os.path.exists(guids_file_path):
    with open(guids_file_path, 'r') as f:
        existing_guids = set(json.load(f))
else:
    existing_guids = set()

# Directory for increment files
increment_dir = 'increment_files'
os.makedirs(increment_dir, exist_ok=True)

# Track the current increment number
increment = 1
existing_files = [f for f in os.listdir(increment_dir) if f.endswith('.json')]
if existing_files:
    increment = max([int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]) + 1

# Add new entries and save them as increment files
for entry in new_entries:
    if entry['guid'] not in existing_guids:
        existing_entries.append(entry)
        existing_guids.add(entry['guid'])
        
        # Save each new entry in an increment file
        increment_file_path = os.path.join(increment_dir, f"{increment}.json")
        with open(increment_file_path, 'w') as f:
            json.dump(entry, f, indent=4)
        increment += 1

# Save the updated list of entries back to the file
with open(file_path, 'w') as f:
    json.dump(existing_entries, f, indent=4)

# Save the updated set of GUIDs back to the GUIDs file
with open(guids_file_path, 'w') as f:
    json.dump(list(existing_guids), f)



def ensure_model_available(model_name):
    """Ensure the model[
        "Researchers", "Data Scientists", "Common People", "Students", "Entrepreneurs",
        "Marketers", "Tech Enthusiasts", "Environmentalists", "Educators", "Healthcare Professionals",
        "Investors", "Content Creators", "Policy Makers", "Journalists", "Travel Enthusiasts",
        "Parents", "Artists", "Fitness Enthusiasts", "Engineers", "Historians",
        "Teachers", "Developers", "Startup Founders", "Writers", "Bloggers",
        "Gamers", "Environmental Activists", "Social Workers", "Consultants", "Small Business Owners",
        "Public Speakers", "Podcasters", "Psychologists", "Sociologists", "Economists",
        "Architects", "Designers", "Photographers", "Lawyers", "Accountants",
        "Athletes", "Personal Trainers", "Chefs", "Food Critics", "Fashion Designers",
        "Musicians", "Film Makers", "Storytellers", "Book Lovers", "Minimalists",
        "Philosophers", "Technologists", "AI Enthusiasts", "Robotics Experts", "Urban Planners",
        "Astronomers", "Mathematicians", "Physicists", "Chemists", "Biologists",
        "Medical Researchers", "Veterinarians", "Farmers", "Futurists", "Cryptocurrency Enthusiasts",
        "HR Professionals", "Recruiters", "Sales Experts", "E-commerce Entrepreneurs", "Digital Nomads",
        "Remote Workers", "Mental Health Advocates", "Mindfulness Coaches", "Public Relations Experts", "Event Planners",
        "Adventure Seekers", "Wildlife Conservationists", "Marine Biologists", "Astronauts", "Space Enthusiasts",
        "DIY Hobbyists", "Car Enthusiasts", "Pet Owners", "Nature Photographers", "Gardeners",
        "Home Decorators", "Interior Designers", "Outdoor Enthusiasts", "Sports Fans", "Social Media Influencers",
        "YouTubers", "Film Critics", "Comedians", "Lifestyle Bloggers", "Relationship Coaches",
        "Spiritual Guides", "Religious Leaders", "Ethicists", "Activists", "Human Rights Advocates",
        "Policy Analysts", "Data Analysts", "Startup Mentors", "Cultural Historians", "Linguists"
    ] is available locally or pull it if missing."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if model_name not in result.stdout:
            print(f"Model {model_name} not found locally. Attempting to pull...")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if pull_result.returncode != 0:
                raise Exception(f"Error pulling model: {pull_result.stderr.strip()}")
            print(f"Model {model_name} pulled successfully.")
        else:
            print(f"Model {model_name} is already available locally.")
    except Exception as e:
        raise Exception(f"Error ensuring model availability: {e}")

def fetch_word_of_the_day():
    """Fetch the Word of the Day from Merriam-Webster."""
    try:
        url = "https://www.merriam-webster.com/word-of-the-day"
        response = subprocess.run(
            ["curl", "-s", "-o", "/tmp/response.html", "-w", "%{http_code}", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if response.stdout.strip() != "200":
            print(f"Failed to fetch Word of the Day. HTTP Status: {response.stdout.strip()}")
            return None
        with open("/tmp/response.html", "r") as file:
            html_content = file.read()
        word = subprocess.run(
            ["grep", "-oP", '(?<=<h2 class="word-header-txt">)[^<]+', "/tmp/response.html"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ).stdout.strip()
        if word:
            print(f"Fetched Word of the Day: {word}")
            return word
        print("Word of the Day not found in the response.")
        return None
    except Exception as e:
        print(f"Error fetching Word of the Day: {e}")
        return None

def generate_random_inputs(topic):
    input_text = topic
    styles = [
        "Origami Artists",
    "Cartographers",
    "Climate Scientists",
    "Geologists",
    "Volcanologists",
    "Storm Chasers",
    "Hurricane Researchers",
    "Tornado Enthusiasts",
    "Urban Historians",
    "Civic Planners",
    "Transportation Designers",
    "Automotive Engineers",
    "Drone Pilots",
    "Space Engineers",
    "Satellite Designers",
    "Rocket Enthusiasts",
    "AI Trainers",
    "Machine Learning Engineers",
    "Cybersecurity Experts",
    "Penetration Testers",
    "Digital Artists",
    "Motion Graphics Designers",
    "Augmented Reality Developers",
    "Holography Experts",
    "3D Printing Specialists",
    "Prototyping Engineers",
    "Industrial Designers",
    "Material Scientists",
    "Nanotechnologists",
    "Biochemists",
    "Geneticists",
    "Microbiologists",
    "Plant Scientists",
    "Agricultural Technologists",
    "Aquaponics Enthusiasts",
    "Permaculture Designers",
    "Ecotourism Experts",
    "Sustainable Architects",
    "Green Builders",
    "Zero Waste Advocates",
    "Upcycling Artists",
    "Circular Economy Specialists",
    "Clean Energy Researchers",
    "Solar Panel Engineers",
    "Wind Energy Specialists",
    "Hydropower Experts",
    "Battery Technologists",
    "Electric Vehicle Enthusiasts",
    "Energy Storage Innovators",
    "Waste Management Experts",
    "Recycling Advocates",
    "Composting Enthusiasts",
    "Environmental Policy Experts",
    "Biodiversity Advocates",
    "Endangered Species Researchers",
    "Wetland Conservationists",
    "Mountain Climbers",
    "Cave Explorers",
    "Arctic Explorers",
    "Desert Survival Experts",
    "Deep Sea Explorers",
    "Polar Researchers",
    "Adventure Photographers",
    "Cultural Storytellers",
    "Folklore Researchers",
    "Traditional Crafts Enthusiasts",
    "Festivals Organizers",
    "Ethnomusicologists",
    "Traditional Dancers",
    "Calligraphers",
    "Typography Designers",
    "Sign Painters",
    "Street Artists",
    "Muralists",
    "Protest Artists",
    "Activist Designers",
    "Political Cartoonists",
    "Social Documentary Filmmakers",
    "War Photographers",
    "Crisis Management Experts",
    "Humanitarian Workers",
    "Refugee Advocates",
    "Disability Activists",
    "Accessible Technology Designers",
    "Inclusive Educators",
    "Interpreters",
    "Translators",
    "Language Preservationists",
    "Dialect Coaches",
    "Cognitive Scientists",
    "Behavioral Economists",
    "Neuroscientists",
    "Psychiatric Researchers",
    "Addiction Counselors",
    "Rehabilitation Experts",
    "Wellness Coaches",
    "Positive Psychologists",
    "Happiness Researchers",
    "Holistic Healers",
         "Archaeologists",
    "Game Designers",
    "Puzzle Enthusiasts",
    "Mythologists",
    "Paleontologists",
    "Astrobiologists",
    "Metaverse Creators",
    "Blockchain Developers",
    "Web3 Enthusiasts",
    "App Developers",
    "Cryptographers",
    "Zoologists",
    "Epidemiologists",
    "Astrologers",
    "Numismatists",
    "Philatelists",
    "Toy Collectors",
    "Historical Reenactors",
    "Craftsmen",
    "Leatherworkers",
    "Blacksmiths",
    "Woodworkers",
    "Beekeepers",
    "Cheesemakers",
    "Winemakers",
    "Brewmasters",
    "Tea Sommeliers",
    "Whiskey Connoisseurs",
    "Luxury Brand Experts",
    "Sneaker Collectors",
    "Fashion Historians",
    "Tattoo Artists",
    "Bodybuilders",
    "Yoga Instructors",
    "Pilates Trainers",
    "Climbers",
    "Parkour Enthusiasts",
    "Survivalists",
    "Bushcrafters",
    "Stunt Performers",
    "Escape Room Designers",
    "Magic Enthusiasts",
    "Illusionists",
    "Stand-Up Comedians",
    "Screenwriters",
    "Playwrights",
    "Voice Actors",
    "Radio Hosts",
    "Audiobook Narrators",
    "Music Producers",
    "Sound Engineers",
    "Songwriters",
    "Lyricists",
    "DJ Artists",
    "Classical Musicians",
    "Orchestra Conductors",
    "Street Performers",
    "Circus Artists",
    "Jugglers",
    "Acrobats",
    "Equestrians",
    "Cyclists",
    "Runners",
    "Marathon Trainers",
    "Triathletes",
    "Swimmers",
    "Scuba Divers",
    "Freedivers",
    "Surfing Enthusiasts",
    "Skiers",
    "Snowboarders",
    "Skaters",
    "Roller Derby Athletes",
    "Mixed Martial Artists",
    "Fencers",
    "Archers",
    "Chess Players",
    "Board Game Designers",
    "Fantasy Writers",
    "Sci-Fi Enthusiasts",
    "Cosplayers",
    "Fanfiction Writers",
    "Graphic Novel Artists",
    "Comic Collectors",
    "Animators",
    "VFX Artists",
    "3D Modelers",
    "Game Streamers",
    "Urban Explorers",
    "Treasure Hunters",
    "Forensic Scientists",
    "Crime Writers",
    "Private Investigators",
    "Ethnographers",
    "Sociolinguists",
    "Educational Technologists",
    "Knowledge Managers",
    "Information Architects",
    "Museum Technologists",
    "Cultural Preservationists",
    "Ecosystem Restorers",
        "Researchers", "Data Scientists", "Common People", "Students", "Entrepreneurs",
        "Marketers", "Tech Enthusiasts", "Environmentalists", "Educators", "Healthcare Professionals",
        "Investors", "Content Creators", "Policy Makers", "Journalists", "Travel Enthusiasts",
        "Parents", "Artists", "Fitness Enthusiasts", "Engineers", "Historians",
        "Teachers", "Developers", "Startup Founders", "Writers", "Bloggers",
        "Gamers", "Environmental Activists", "Social Workers", "Consultants", "Small Business Owners",
        "Public Speakers", "Podcasters", "Psychologists", "Sociologists", "Economists",
        "Architects", "Designers", "Photographers", "Lawyers", "Accountants",
        "Athletes", "Personal Trainers", "Chefs", "Food Critics", "Fashion Designers",
        "Musicians", "Film Makers", "Storytellers", "Book Lovers", "Minimalists",
        "Philosophers", "Technologists", "AI Enthusiasts", "Robotics Experts", "Urban Planners",
        "Astronomers", "Mathematicians", "Physicists", "Chemists", "Biologists",
        "Medical Researchers", "Veterinarians", "Farmers", "Futurists", "Cryptocurrency Enthusiasts",
        "HR Professionals", "Recruiters", "Sales Experts", "E-commerce Entrepreneurs", "Digital Nomads",
        "Remote Workers", "Mental Health Advocates", "Mindfulness Coaches", "Public Relations Experts", "Event Planners",
        "Adventure Seekers", "Wildlife Conservationists", "Marine Biologists", "Astronauts", "Space Enthusiasts",
        "DIY Hobbyists", "Car Enthusiasts", "Pet Owners", "Nature Photographers", "Gardeners",
        "Home Decorators", "Interior Designers", "Outdoor Enthusiasts", "Sports Fans", "Social Media Influencers",
        "YouTubers", "Film Critics", "Comedians", "Lifestyle Bloggers", "Relationship Coaches",
        "Spiritual Guides", "Religious Leaders", "Ethicists", "Activists", "Human Rights Advocates",
        "Policy Analysts", "Data Analysts", "Startup Mentors", "Cultural Historians", "Linguists"
    ]
    no_words = 50000
    blog_style = random.choice(styles)
    return input_text, no_words, blog_style
class AICrew:
    """AI Crew with specialized roles."""
    def __init__(self, model_name):
        self.model_name = model_name
    def run_ollama(self, prompt):
        """Run a prompt through the Ollama model."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"Error running Ollama: {result.stderr.strip()}")
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"
    
    def creative_write(self, topic):
        prompt = f"""
        Write an engaging blog post about the topic: {topic}. 
        Include interesting anecdotes, relatable examples, and a strong conclusion.
        """
        return self.run_ollama(prompt)

    def fact_check(self, content):
        prompt = f"""
        Verify the accuracy of the following text: {content}. 
        Highlight any inaccuracies and provide confidence levels for each fact.
        """
        return self.run_ollama(prompt)

    def optimize_for_seo(self, content, keyword):
        prompt = f"""
        Optimize the following blog post for SEO. 
        Improve keyword density, add meta descriptions, and ensure readability.
        Content: {content}
        """
        return self.run_ollama(prompt)

    def edit_content(self, content):
        prompt = f"""
        Edit the following blog post for tone, grammar, and readability. 
        Make it polished and professional. Content: {content}
        """
        return self.run_ollama(prompt)
def get_ollama_response(input_text, no_words, blog_style, word_of_the_day, model_name="llama3"):
    """Generate a blog using Ollama with the provided inputs."""
    today_year = datetime.now().year
    prompts = [
        # Conversational Prompt
        f"""
            Write a blog for {blog_style} professionals about the topic "{input_text}". Make it engaging and easy to read, with approximately {no_words} words. 
            Use the word "{word_of_the_day}" creatively and naturally. Include:
            - Practical examples or relatable anecdotes to illustrate key points.
            - Synonyms and alternative phrases for "{input_text}" to improve SEO.
            - Eye-catching headings and bullet points for better readability.
            - A concluding call-to-action to inspire further thought or action.with seo optimize
        """,

        # Research-Oriented Prompt
        f"""
            Develop an in-depth, SEO-optimized blog for {blog_style} professionals on the topic "{input_text}". The blog should:
            - Be around {no_words} words.
            - Use "{word_of_the_day}" thoughtfully in a relevant context.
            - Include statistics, references, or data-driven insights to support the content.
            - Highlight challenges, solutions, or innovations related to the topic.
            - Use structured formatting, with headings, subheadings, and lists to organize the information.with seo optimize
        """,

        # Storytelling Prompt
        f"""
            Create a narrative-style blog for {blog_style} professionals that focuses on the topic "{input_text}". The blog should:
            - Be {no_words} words long.
            - Include the word "{word_of_the_day}" in a meaningful and creative way.
            - Start with a captivating story or anecdote to draw readers in.
            - Include a moral, takeaway, or lesson that ties back to the topic.
            - Use vivid language and relatable examples to make the content engaging.with seo optimize
        """,

        # Listicle Prompt
        f"""
            Write a blog for {blog_style} professionals in the form of a listicle about "{input_text}". The blog should:
            - Contain {no_words} words and use the word "{word_of_the_day}" naturally.
            - Include at least 5 main points or items, each with a brief explanation or example.
            - Use headings and bullet points to make the content scannable.
            - End with a summary or call-to-action to encourage further engagement. with seo optimize
        """,

        # Opinion Piece Prompt
        f"""
            Write an opinion-based blog for {blog_style} professionals about the topic "{input_text}". The blog should:
            - Be {no_words} words long.
            - Use "{word_of_the_day}" in a bold and creative way to emphasize your perspective.
            - Present a clear stance on the topic and support it with logical arguments.
            - Address potential counterarguments and provide rebuttals.
            - Conclude with a strong and memorable statement to leave an impact.with seo optimize
        """,

        # How-To Guide Prompt
        f"""
            Create a detailed how-to guide for {blog_style} professionals about "{input_text}". The blog should:
            - Be {no_words} words in length and include "{word_of_the_day}" in a practical context.
            - Outline clear, step-by-step instructions with examples or tips.
            - Use headings, subheadings, and numbered lists for easy navigation.
            - Address common challenges and provide solutions for each step.with seo optimize
        """,

        # FAQ-Style Prompt
        f"""
            Write a blog for {blog_style} professionals in a FAQ format about the topic "{input_text}". The blog should:
            - Be around {no_words} words long and include "{word_of_the_day}" in a creative way.
            - Address at least 5 common questions or concerns related to the topic.
            - Provide concise and informative answers with actionable advice.
            - Use headings or bold text for each question to improve readability.with seo optimize
        """,

        # Problem-Solution Prompt
        f"""
            Write a blog for {blog_style} professionals focused on solving a specific problem related to "{input_text}". The blog should:
            - Be {no_words} words long.
            - Use "{word_of_the_day}" in the context of addressing or overcoming the problem.
            - Clearly define the problem and explain why it matters.
            - Offer practical solutions or strategies to tackle the issue.
            - Conclude with a summary or call-to-action encouraging readers to act.with seo optimize
        """,

        # Analytical Prompt
        f"""
            Create an analytical blog for {blog_style} professionals that examines "{input_text}". The blog should:
            - Be approximately {no_words} words in length.
            - Include "{word_of_the_day}" to emphasize key points or findings.
            - Break down the topic into smaller components or trends for analysis.
            - Use data, graphs, or references to support arguments.
            - Conclude with insights or predictions based on the analysis.with seo optimize
        """,

        # Creative Prompt
        f"""
            Write a creative blog for {blog_style} professionals about the topic "{input_text}". The blog should:
            - Be around {no_words} words and incorporate "{word_of_the_day}" in an imaginative way.
            - Use storytelling, metaphors, or fictional scenarios to explore the topic.
            - Focus on originality and unique perspectives to captivate readers.
            - Provide a meaningful takeaway or lesson at the end.with seo optimize
        """,
        f"write a title  of the blog post with 60 characters  and Write a blog {blog_style}: Why {input_text} No Longer Works in {today_year} and What to Do Instead. Be around {no_words} words and include '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {blog_style} Professionals Can Benefit from {input_text} in {today_year}. Write {no_words} words and creatively use '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog 5 Underrated Tools for {blog_style} Professionals to Master {input_text} in {today_year}. Cover this in {no_words} words with '{word_of_the_day}' included.",
        f"write a title  of the blog post with 60 characters  and Write a blog What {input_text} Means for {blog_style} Professionals in {today_year}. Analyze in {no_words} words while emphasizing '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {blog_style} Professionals Can Master {input_text} in {today_year}. Provide actionable tips in {no_words} words and include '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog Why {input_text} Is Critical for {blog_style} Success in {today_year}. Dive into the topic in {no_words} words, using '{word_of_the_day}' naturally.",
        f"write a title  of the blog post with 60 characters  and Write a blog Lessons on {input_text} for {blog_style} Professionals in {today_year}. Share insights in {no_words} words, creatively using '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Evolution of {input_text} for {blog_style} Professionals in {today_year}. Cover its journey in {no_words} words and include '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {blog_style} Professionals Can Overcome Challenges with {input_text} in {today_year}. Offer solutions in {no_words} words, using '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog Behind the Scenes: {input_text} for {blog_style} in {today_year}. Unveil insights in {no_words} words with a focus on '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog Top Strategies for {blog_style} Professionals to Excel in {input_text} in {today_year}. Outline in {no_words} words while incorporating '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog 5 Ways {blog_style} Professionals Can Leverage {input_text} in {today_year}. Highlight approaches in {no_words} words, using '{word_of_the_day}' creatively.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Role of {input_text} in {blog_style} Growth in {today_year}. Explain in {no_words} words while weaving in '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog Why {input_text} Matters for {blog_style} in {today_year}. Discuss its significance in {no_words} words and incorporate '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Future of {input_text} for {blog_style} Professionals Beyond {today_year}. Speculate in {no_words} words, making '{word_of_the_day}' central to the narrative.",
        f"write a title  of the blog post with 60 characters  and Write a blog 5 Key Insights on {input_text} for {blog_style} in {today_year}. Provide them in {no_words} words and include '{word_of_the_day}' creatively.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {input_text} Is Shaping the Future of {blog_style} in {today_year}. Write {no_words} words and make '{word_of_the_day}' a focus point.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Importance of {input_text} for {blog_style} Professionals in {today_year}. Highlight why in {no_words} words, incorporating '{word_of_the_day}' naturally.",
        f"write a title  of the blog post with 60 characters  and Write a blog How to Use {input_text} Effectively as a {blog_style} Professional in {today_year}. Cover this in {no_words} words, using '{word_of_the_day}' practically.",
        f"The Challenges of {input_text} in {blog_style} and How to Overcome Them in {today_year}. Write {no_words} words and creatively use '{word_of_the_day}'.",
        f"How {blog_style} Professionals Can Navigate {input_text} in {today_year}. Provide guidance in {no_words} words and include '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog 5 Lessons Learned from {input_text} for {blog_style} Professionals in {today_year}. Share these in {no_words} words while incorporating '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Power of {input_text} for {blog_style} in {today_year}. Explain its influence in {no_words} words, ensuring '{word_of_the_day}' is highlighted.",
        f"write a title  of the blog post with 60 characters  and Write a blog Why {input_text} Should Be a Priority for {blog_style} in {today_year}. Discuss in {no_words} words while weaving '{word_of_the_day}' into the narrative.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {blog_style} Professionals Can Improve Their Work Through {input_text} in {today_year}. Cover this in {no_words} words and use '{word_of_the_day}'.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Impact of {input_text} on {blog_style} Success in {today_year}. Analyze in {no_words} words while making '{word_of_the_day}' central to the discussion.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Role of {input_text} in Transforming {blog_style} in {today_year}. Write about this in {no_words} words, ensuring '{word_of_the_day}' is included.",
        f"write a title  of the blog post with 60 characters  and Write a blog How {input_text} Can Revolutionize {blog_style} in {today_year}. Explore the potential in {no_words} words with '{word_of_the_day}' creatively used.",
        f"write a title  of the blog post with 60 characters  and Write a blog The Ultimate Guide to {input_text} for {blog_style} Professionals in {today_year}. Write this in {no_words} words, incorporating '{word_of_the_day}' throughout."


    ]                     
    prompt = random.choice(prompts) + " Use a professional yet approachable tone, and make sure the content is easy to read, with clear subheadings and varied sentence structure for improved readability. Integrate keywords relevant to the tool and field"
    try:
        ensure_model_available(model_name)
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Error running model: {result.stderr.strip()}")
        print("Ollama Response Retrieved Successfully.")
        crew = AICrew(model_name)
        final_content = crew.edit_content(result.stdout.strip())
        promptTitle = f"what is the title  {final_content}?"
        resultTitle = subprocess.run(
            ["ollama", "run", model_name, promptTitle],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {
            "blog": (final_content),
            "title": resultTitle.stdout.strip()
        }
    except Exception as e:
        return f"Error during query: {e}"

def send_email(recipient_email, subject, content, attachment_path):
    """Send the generated blog content via email."""
    try:
        sender_email = "ooseims@gmail.com"
        sender_password = "cbnq ykhu jhjn rxkp"

        # Set up the MIME message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject.replace("\n", " ").strip()

        # Attach the blog content
        message.attach(MIMEText(content, "plain"))
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(attachment_path)}",
            )
            message.attach(part)
        # Connect to the SMTP server
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            print(f"Email sent successfully to {recipient_email}.")
    except Exception as e:
        print(f"Error sending email: {e}")
    
word_of_the_day = fetch_word_of_the_day() or "innovation"
try:
    for entry in existing_entries:
        __title = entry["title"]
        image_url = f"https://pollinations.ai/p/{__title}?width={width}&height={height}&seed={seed}&model={model}"
        file_name = f"image.jpg"
        file_name1 = f"generated/image_{seed}.jpg"
        try:
            download_image(image_url, file_name)
            download_image(image_url, file_name1)
        except Exception as e:
            print(f"Error generating random inputs for entry '{entry['title']}': {e}")
            continue  # Skip to the next entry if input generation fails
        
        # Save the updated seed
        with open(seed_file, 'w') as file:
            file.write(str(seed))
        try:
            # Generate random inputs based on the entry's title and description
            topic, word_count, audience = generate_random_inputs(entry["title"] + " " + entry["description"])
        except Exception as e:
            print(f"Error generating random inputs for entry '{entry['title']}': {e}")
            continue  # Skip to the next entry if input generation fails
    
        try:
            # Generate blog content using the topic, word count, audience, and word of the day
            blog_content = get_ollama_response(topic, word_count, audience, word_of_the_day)
        except Exception as e:
            print(f"Error generating blog content for entry '{entry['title']}': {e}")
            continue  # Skip to the next entry if content generation fails
    
        print("Generated Blog Content:\n")
        print(blog_content)
    
        try:
            # Check if the content contains "Error"
            if "Error" not in blog_content:
                # Send the blog content via email
                send_email(
                    recipient_email="edwardlorilla2220.edwardlancelorilla@blogger.com",
                    subject=blog_content["title"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace('The title of the blog post is', '').replace(':', '').replace('<br>', '').replace('The title of this edited blog post is', '').replace('Based on your edited blog post, I would title it', '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', ''),
                    content=blog_content["blog"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace(':', '').replace('Let me know if you need any further assistance', '').replace('"', '').replace('<br>', '').replace("Here's a polished and professional version of the blog post", '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('\n', '<br>').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', ''),
                    attachment_path=file_name
                )
                title = blog_content["title"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace('The title of the blog post is', '').replace(':', '').replace('<br>', '').replace('The title of this edited blog post is', '').replace('Based on your edited blog post, I would title it', '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', '')
                summary = blog_content["blog"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace(':', '').replace('Let me know if you need any further assistance', '').replace('"', '').replace('<br>', '').replace("Here's a polished and professional version of the blog post", '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('\n', '<br>').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', '')
                content = blog_content["blog"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace(':', '').replace('Let me know if you need any further assistance', '').replace('"', '').replace('<br>', '').replace("Here's a polished and professional version of the blog post", '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('\n', '<br>').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', '')
                keywords = "SEO, website, marketing, search engines"
                slug = blog_content["title"].replace('"', '').replace("Here's the polished and professional version of the blog post", '').replace('The title of the blog post is', '').replace(':', '').replace('<br>', '').replace('The title of this edited blog post is', '').replace('Based on your edited blog post, I would title it', '').replace('Here is the edited blog post', '').replace('Here is the revised blog post', '').replace('The title is', '').replace('The title of this blog post is', '').replace('Here is a polished and professional version of the blog post', '')
                thumbnail = "default-thumbnail.jpg" 
                
                # Insert the generated content into the MySQL database
                insert_blog_post_to_db(title, summary, content, keywords, slug, thumbnail)
            else:
                print(f"Blog content generation failed for entry '{entry['title']}'. Email will not be sent.")
        except Exception as e:
            print(f"Error sending email for entry '{entry['title']}': {e}")
except Exception as e:
    print(f"Error generating random inputs for entry: {e}")
    
    
    
        
        
        
        
        
        
        
        
        
        
        
