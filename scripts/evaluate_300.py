import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from tqdm import tqdm

test_data = [
    # Ad Hominem (38 examples)
    {"text": "Why should we listen to your views on ethics? You were arrested once.", "true_label": "ad_hominem"},
    {"text": "Of course you want a raise; you're just greed personified.", "true_label": "ad_hominem"},
    {"text": "He's just a politician; you can't believe a word he says.", "true_label": "ad_hominem"},
    {"text": "Your argument is invalid because you're a hypocrite.", "true_label": "ad_hominem"},
    {"text": "Don't listen to him, he's from a family of criminals.", "true_label": "ad_hominem"},
    {"text": "You only support this project because you're hoping to get a promotion.", "true_label": "ad_hominem"},
    {"text": "Why believe a doctor who is obviously overweight when he talks about health?", "true_label": "ad_hominem"},
    {"text": "She's too young to understand how the world really works.", "true_label": "ad_hominem"},
    {"text": "He's just an ivory tower academic with no real-world experience.", "true_label": "ad_hominem"},
    {"text": "Of course you'd say that, you're a member of that political party.", "true_label": "ad_hominem"},
    {"text": "Your opinion on this matter is irrelevant because of your religious background.", "true_label": "ad_hominem"},
    {"text": "Don't take advice from someone who couldn't even finish college.", "2rue_label": "ad_hominem"}, # Typo in label key (fixed below)
    {"text": "He's a known liar, so his data must be wrong.", "true_label": "ad_hominem"},
    {"text": "You're only defensive because you're insecure.", "true_label": "ad_hominem"},
    {"text": "Why listen to a billionaire talk about poverty?", "true_label": "ad_hominem"},
    {"text": "She's just a disgruntled former employee; don't trust her.", "true_label": "ad_hominem"},
    {"text": "He's too old and out of touch to have a valid opinion.", "true_label": "ad_hominem"},
    {"text": "You're just saying that because you're a man/woman.", "true_label": "ad_hominem"},
    {"text": "His argument is weak because he has a history of mental health issues.", "true_label": "ad_hominem"},
    {"text": "Why follow the advice of a failed businessman?", "true_label": "ad_hominem"},
    {"text": "She's just trying to get attention with these radical ideas.", "true_label": "ad_hominem"},
    {"text": "He's a puppet for the corporate elites.", "true_label": "ad_hominem"},
    {"text": "You're just biased because of where you grew up.", "true_label": "ad_hominem"},
    {"text": "Don't listen to her, she's obsessed with conspiracy theories.", "true_label": "ad_hominem"},
    {"text": "His logic is flawed because he's fundamentally a bad person.", "true_label": "ad_hominem"},
    {"text": "You're only arguing this point to make yourself look smart.", "true_label": "ad_hominem"},
    {"text": "He's just a shill for the pharmaceutical companies.", "true_label": "ad_hominem"},
    {"text": "Why trust someone who has been divorced three times with marriage advice?", "true_label": "ad_hominem"},
    {"text": "She's just a celebrity trying to stay relevant.", "true_label": "ad_hominem"},
    {"text": "He's a known socialist, so his economic plan is garbage.", "true_label": "ad_hominem"},
    {"text": "You're just a kid, you don't know anything about taxes.", "true_label": "ad_hominem"},
    {"text": "His credentials are fake, so his theory is wrong.", "true_label": "ad_hominem"},
    {"text": "You're only saying that because you're part of the system.", "true_label": "ad_hominem"},
    {"text": "Don't listen to him, he's just an angry old man.", "true_label": "ad_hominem"},
    {"text": "She's just a puppet for her husband's interests.", "true_label": "ad_hominem"},
    {"text": "He's a criminal, so anything he says is a lie.", "true_label": "ad_hominem"},
    {"text": "You're just jealous of my success, that's why you're critical.", "true_label": "ad_hominem"},
    {"text": "He's a loudmouth who just wants to hear himself talk.", "true_label": "ad_hominem"},

    # Appeal to Authority (38 examples)
    {"text": "This cereal is healthy because an Olympic athlete says so.", "true_label": "appeal_to_authority"},
    {"text": "My doctor said we should invest in gold, so I'm doing it.", "true_label": "appeal_to_authority"},
    {"text": "The Pope said this scientific theory is correct, so it must be.", "true_label": "appeal_to_authority"},
    {"text": "A famous actor says this skin cream works wonders.", "true_label": "appeal_to_authority"},
    {"text": "The principal says we must eat more spinach, so we should.", "true_label": "appeal_to_authority"},
    {"text": "Because the CEO of the tech company said so, the new software is perfect.", "true_label": "appeal_to_authority"},
    {"text": "My history teacher told me that the sky was green in 1800, so it must be true.", "true_label": "appeal_to_authority"},
    {"text": "A Nobel Prize winner in Physics says this diet is the best.", "true_label": "appeal_to_authority"},
    {"text": "The police chief said the new law is good for the community.", "true_label": "appeal_to_authority"},
    {"text": "A popular TV host recommended this supplement, so it must be effective.", "true_label": "appeal_to_authority"},
    {"text": "My grandfather, who lived to 100, said to always eat raw onions.", "true_label": "appeal_to_authority"},
    {"text": "The King said the potion would cure the plague, so everyone drank it.", "true_label": "appeal_to_authority"},
    {"text": "A famous designer says these shoes are the future of fashion.", "true_label": "appeal_to_authority"},
    {"text": "The captain of the football team said this is the best way to study.", "true_label": "appeal_to_authority"},
    {"text": "A reality star says this cryptocurrency is going to moon.", "true_label": "appeal_to_authority"},
    {"text": "The governor said this bridge is safe, so don't worry about the cracks.", "true_label": "appeal_to_authority"},
    {"text": "An expert in ancient history says this modern car is a masterpiece.", "true_label": "appeal_to_authority"},
    {"text": "My favorite YouTuber says this energy drink increases focus.", "true_label": "appeal_to_authority"},
    {"text": "The general said this war is necessary for peace.", "true_label": "appeal_to_authority"},
    {"text": "A famous chef says this microwave dinner is gourmet quality.", "true_label": "appeal_to_authority"},
    {"text": "The president of the bank says the economy is booming.", "true_label": "appeal_to_authority"},
    {"text": "A chess grandmaster says this political candidate is the best choice.", "true_label": "appeal_to_authority"},
    {"text": "My yoga instructor said I should avoid gluten for better mental health.", "true_label": "appeal_to_authority"},
    {"text": "A tech billionaire says we'll be living on Mars by 2030.", "true_label": "appeal_to_authority"},
    {"text": "The school board president said the new textbooks are unbiased.", "true_label": "appeal_to_authority"},
    {"text": "A famous singer says this meditation app changed her life.", "true_label": "appeal_to_authority"},
    {"text": "The head of the labor union said this contract is the best possible deal.", "true_label": "appeal_to_authority"},
    {"text": "An astronaut said the earth is flat, so now I believe him.", "true_label": "appeal_to_authority"},
    {"text": "The fashion editor of a top magazine says neon is back in style.", "true_label": "appeal_to_authority"},
    {"text": "A pro gamer says this keyboard will make you a better player.", "true_label": "appeal_to_authority"},
    {"text": "The lead singer of that band says we should support this cause.", "true_label": "appeal_to_authority"},
    {"text": "My boss said this strategy is foolproof, so I won't question it.", "true_label": "appeal_to_authority"},
    {"text": "A famous architect says this building is a masterpiece of design.", "true_label": "appeal_to_authority"},
    {"text": "The mayor said the new park will solve all our traffic problems.", "true_label": "appeal_to_authority"},
    {"text": "An influencer with 10 million followers says this tea helps you lose weight.", "true_label": "appeal_to_authority"},
    {"text": "The local priest said that reading fiction is a sin.", "true_label": "appeal_to_authority"},
    {"text": "A former spy says this lock is unbreakable.", "true_label": "appeal_to_authority"},
    {"text": "The coach said we must win the game to be happy.", "true_label": "appeal_to_authority"},

    # Bandwagon (38 examples)
    {"text": "Everyone is buying this iPhone, so it must be the best.", "true_label": "bandwagon"},
    {"text": "Millions of people can't be wrong about this movie.", "true_label": "bandwagon"},
    {"text": "Most Australians believe in this law, so it's a good one.", "true_label": "bandwagon"},
    {"text": "All my friends are on TikTok, so I should be too.", "true_label": "bandwagon"},
    {"text": "The whole world knows that this is the best brand of shoes.", "true_label": "bandwagon"},
    {"text": "Jump on the bandwagon and support the winning team!", "true_label": "bandwagon"},
    {"text": "Everyone is going to this party, you should come too.", "true_label": "bandwagon"},
    {"text": "Nobody uses landlines anymore, so they're completely useless.", "true_label": "bandwagon"},
    {"text": "The majority of students voted for this change, so it's obviously right.", "true_label": "bandwagon"},
    {"text": "Everyone is investing in this stock, don't miss out!", "true_label": "bandwagon"},
    {"text": "This diet is popular with all the celebrities, so it must work.", "true_label": "bandwagon"},
    {"text": "The most-watched show on Netflix is always the highest quality.", "true_label": "bandwagon"},
    {"text": "Everyone I know thinks he's guilty, so he must be.", "true_label": "bandwagon"},
    {"text": "This new fashion trend is everywhere, you need to follow it.", "true_label": "bandwagon"},
    {"text": "The entire neighborhood is using this landscaping service.", "true_label": "bandwagon"},
    {"text": "Everyone is switching to solar power, so you should too.", "true_label": "bandwagon"},
    {"text": "The consensus among all my colleagues is that we should quit.", "true_label": "bandwagon"},
    {"text": "Everyone says this restaurant is amazing, so I'm sure I'll love it.", "true_label": "bandwagon"},
    {"text": "All the cool kids are wearing these sneakers.", "true_label": "bandwagon"},
    {"text": "Everyone is talking about this new book, it must be a masterpiece.", "true_label": "bandwagon"},
    {"text": "The majority of the population supports this initiative.", "true_label": "bandwagon"},
    {"text": "Everyone is downloading this app right now.", "true_label": "bandwagon"},
    {"text": "Everyone knows that cats are better than dogs.", "true_label": "bandwagon"},
    {"text": "The crowd was cheering for the blue team, so they must be the favorites.", "true_label": "bandwagon"},
    {"text": "Everyone is complaining about the new schedule.", "true_label": "bandwagon"},
    {"text": "The trendiest influencers are all using this filters.", "true_label": "bandwagon"},
    {"text": "Everyone is worried about the future, so you should be too.", "true_label": "bandwagon"},
    {"text": "The most common opinion is usually the correct one.", "true_label": "bandwagon"},
    {"text": "Everyone is avoiding that part of town.", "true_label": "bandwagon"},
    {"text": "Everyone is saying the same thing, so there must be some truth to it.", "true_label": "bandwagon"},
    {"text": "The whole world is watching this event.", "true_label": "bandwagon"},
    {"text": "Everyone is signing the petition, so you should sign it too.", "true_label": "bandwagon"},
    {"text": "The popular vote shows that people want this law.", "true_label": "bandwagon"},
    {"text": "Everyone is buying air purifiers these days.", "true_label": "bandwagon"},
    {"text": "Everyone is obsessed with this new TV series.", "true_label": "bandwagon"},
    {"text": "The consensus is that we need a change in leadership.", "true_label": "bandwagon"},
    {"text": "Everyone is switching to this new internet provider.", "true_label": "bandwagon"},
    {"text": "Everyone believes that hard work leads to success.", "true_label": "bandwagon"},

    # False Dilemma (37 examples)
    {"text": "Either you support the war, or you are a traitor.", "true_label": "false_dilemma"},
    {"text": "We either increase the budget or we fail completely.", "true_label": "false_dilemma"},
    {"text": "You are either with us or against us.", "true_label": "false_dilemma"},
    {"text": "Either fix the car or sell it for scrap.", "true_label": "false_dilemma"},
    {"text": "We must either raise taxes or cut health services.", "true_label": "false_dilemma"},
    {"text": "Either you love me or you hate me.", "true_label": "false_dilemma"},
    {"text": "We either go to the beach or stay home and be miserable.", "true_label": "false_dilemma"},
    {"text": "Either you're a genius or you're an idiot.", "true_label": "false_dilemma"},
    {"text": "We either ban all plastic or destroy the planet.", "true_label": "false_dilemma"},
    {"text": "Either you're part of the solution or part of the problem.", "true_label": "false_dilemma"},
    {"text": "We either invest in this company or lose all our money.", "true_label": "false_dilemma"},
    {"text": "Either you agree with me or you're wrong.", "true_label": "false_dilemma"},
    {"text": "We either buy a new house or live in this dump forever.", "true_label": "false_dilemma"},
    {"text": "Either you help me or you're my enemy.", "true_label": "false_dilemma"},
    {"text": "We either pass this law or the country will descend into chaos.", "true_label": "false_dilemma"},
    {"text": "Either you're a patriot or you're a criminal.", "true_label": "false_dilemma"},
    {"text": "We either win this game or our season is over.", "true_label": "false_dilemma"},
    {"text": "Either you work hard or you'll be a failure for life.", "true_label": "false_dilemma"},
    {"text": "We either change our habits or face extinction.", "true_label": "false_dilemma"},
    {"text": "Either you're a leader or a follower.", "true_label": "false_dilemma"},
    {"text": "We either lower prices or lose all our customers.", "true_label": "false_dilemma"},
    {"text": "Either you eat your vegetables or you don't get dessert.", "true_label": "false_dilemma"},
    {"text": "We either support the arts or lose our culture.", "true_label": "false_dilemma"},
    {"text": "Either you're a winner or a loser.", "true_label": "false_dilemma"},
    {"text": "We either fight back or surrender.", "true_label": "false_dilemma"},
    {"text": "Either you're a believer or an atheist.", "true_label": "false_dilemma"},
    {"text": "We either build the wall or suffer from crime.", "true_label": "false_dilemma"},
    {"text": "Either you're a liberal or a conservative.", "true_label": "false_dilemma"},
    {"text": "We either save the whales or let the ocean die.", "true_label": "false_dilemma"},
    {"text": "Either you're a success or a disappointment.", "true_label": "false_dilemma"},
    {"text": "We either adapt or perish.", "true_label": "false_dilemma"},
    {"text": "Either you're a morning person or a night owl.", "true_label": "false_dilemma"},
    {"text": "We either expand the city or face overcrowding.", "true_label": "false_dilemma"},
    {"text": "Either you're a cat person or a dog person.", "true_label": "false_dilemma"},
    {"text": "We either increase security or face more attacks.", "true_label": "false_dilemma"},
    {"text": "Either you're a fast learner or you'll never get it.", "true_label": "false_dilemma"},
    {"text": "We either fix the roof or the whole house will rot.", "true_label": "false_dilemma"},

    # Hasty Generalization (37 examples)
    {"text": "My grandfather smoked and lived to be 100, so smoking is fine.", "true_label": "hasty_generalization"},
    {"text": "I met one rude French person, so all French people are rude.", "true_label": "hasty_generalization"},
    {"text": "This pit bull bit a child, therefore all pit bulls are dangerous.", "true_label": "hasty_generalization"},
    {"text": "I failed my first test, so I'm going to fail the whole course.", "true_label": "hasty_generalization"},
    {"text": "He wore a red shirt today and he's mean, so people in red shirts are mean.", "true_label": "hasty_generalization"},
    {"text": "I saw a bird fly south, so winter must be here tomorrow.", "true_label": "hasty_generalization"},
    {"text": "One person got sick from the food, so the entire menu is poison.", "true_label": "hasty_generalization"},
    {"text": "I had a bad experience at that store, so I'll never shop there again.", "true_label": "hasty_generalization"},
    {"text": "The first book in the series was great, so clearly they all are.", "true_label": "hasty_generalization"},
    {"text": "I saw a politician lie, so all politicians are liars.", "true_label": "hasty_generalization"},
    {"text": "My neighbor's car broke down, so that brand of car is junk.", "true_label": "hasty_generalization"},
    {"text": "I met a quiet student, so all students must be introverts.", "true_label": "hasty_generalization"},
    {"text": "The first five minutes of the movie were boring, so the whole movie is bad.", "true_label": "hasty_generalization"},
    {"text": "I saw a few drops of rain, so a thunderstorm is coming.", "true_label": "hasty_generalization"},
    {"text": "One person won the lottery, so anybody can win it easily.", "true_label": "hasty_generalization"},
    {"text": "I saw a teenager littering, so all teenagers are irresponsible.", "true_label": "hasty_generalization"},
    {"text": "The first time I tried sushi I didn't like it, so I hate all Japanese food.", "true_label": "hasty_generalization"},
    {"text": "I saw a homeless person drinking, so all homeless people are alcoholics.", "true_label": "hasty_generalization"},
    {"text": "The first chapter was easy, so the whole book will be a breeze.", "true_label": "hasty_generalization"},
    {"text": "I saw a car with a flat tire, so that model has bad tires.", "true_label": "hasty_generalization"},
    {"text": "I met a friendly cat, so all cats are friendly.", "true_label": "hasty_generalization"},
    {"text": "The first person I met from that city was helpful, so everyone there must be nice.", "true_label": "hasty_generalization"},
    {"text": "I saw one person succeed without a degree, so college is useless for everyone.", "true_label": "hasty_generalization"},
    {"text": "I had a headache after eating chocolate, so chocolate causes headaches for everyone.", "true_label": "hasty_generalization"},
    {"text": "I saw a dog barking at a mailbox, so all dogs hate mailmen.", "true_label": "hasty_generalization"},
    {"text": "The first day of work was stressful, so this job is going to be a nightmare.", "true_label": "hasty_generalization"},
    {"text": "I saw a cyclist run a red light, so all cyclists are lawbreakers.", "true_label": "hasty_generalization"},
    {"text": "I heard one bad song by that band, so they must be a terrible band.", "true_label": "hasty_generalization"},
    {"text": "I saw a child throw a tantrum, so all children are spoiled.", "true_label": "hasty_generalization"},
    {"text": "The first flight I took was delayed, so that airline is always late.", "true_label": "hasty_generalization"},
    {"text": "I saw a spider in my room, so the whole house is infested with spiders.", "true_label": "hasty_generalization"},
    {"text": "I met a shy actor, so all actors are actually introverts.", "true_label": "hasty_generalization"},
    {"text": "The first episode was great, so the whole series is must-watch TV.", "true_label": "hasty_generalization"},
    {"text": "I saw a person with tattoos acting rudely, so people with tattoos are all rude.", "true_label": "hasty_generalization"},
    {"text": "I had a bad meal at a restaurant, so the chef must be incompetent.", "true_label": "hasty_generalization"},
    {"text": "I saw a driver texting, so all drivers on the road are distracted.", "true_label": "hasty_generalization"},
    {"text": "I met a person who likes pineapple on pizza, so clearly everyone loves it.", "true_label": "hasty_generalization"},

    # No Fallacy (37 examples)
    {"text": "Water boils at 100 degrees Celsius at sea level.", "true_label": "no_fallacy"},
    {"text": "The sun rises in the east.", "true_label": "no_fallacy"},
    {"text": "According to the census, the population has grown by 10%.", "true_label": "no_fallacy"},
    {"text": "I need to go to the store to buy milk.", "true_label": "no_fallacy"},
    {"text": "The earth orbits the sun.", "true_label": "no_fallacy"},
    {"text": "If it rains, the grass gets wet.", "true_label": "no_fallacy"},
    {"text": "I'm feeling tired because I didn't sleep well last night.", "true_label": "no_fallacy"},
    {"text": "The movie starts at 7 PM tonight.", "true_label": "no_fallacy"},
    {"text": "I'm wearing a blue shirt today.", "true_label": "no_fallacy"},
    {"text": "The bridge was built in 1950.", "true_label": "no_fallacy"},
    {"text": "The cat is sleeping on the sofa.", "true_label": "no_fallacy"},
    {"text": "I enjoy listening to jazz music.", "true_label": "no_fallacy"},
    {"text": "The library is closed on Sundays.", "true_label": "no_fallacy"},
    {"text": "I have two sisters and one brother.", "true_label": "no_fallacy"},
    {"text": "The capital of France is Paris.", "true_label": "no_fallacy"},
    {"text": "It's cold outside, so you should wear a coat.", "true_label": "no_fallacy"},
    {"text": "I'm learning how to play the piano.", "true_label": "no_fallacy"},
    {"text": "The train was late due to a signal failure.", "true_label": "no_fallacy"},
    {"text": "I'm going to cook dinner now.", "true_label": "no_fallacy"},
    {"text": "The flowers are blooming in the garden.", "true_label": "no_fallacy"},
    {"text": "I need to finish this report by Friday.", "true_label": "no_fallacy"},
    {"text": "The sky is blue on a clear day.", "true_label": "no_fallacy"},
    {"text": "I'm grateful for your help.", "true_label": "no_fallacy"},
    {"text": "The car needs an oil change.", "true_label": "no_fallacy"},
    {"text": "I like to go for a run in the morning.", "true_label": "no_fallacy"},
    {"text": "The restaurant is famous for its pizza.", "true_label": "no_fallacy"},
    {"text": "I'm reading a very interesting book.", "true_label": "no_fallacy"},
    {"text": "The phone is ringing.", "true_label": "no_fallacy"},
    {"text": "I'm planning a trip to Japan.", "true_label": "no_fallacy"},
    {"text": "The computer is running slowly.", "true_label": "no_fallacy"},
    {"text": "I need to recharge my battery.", "true_label": "no_fallacy"},
    {"text": "The coffee is too hot to drink.", "true_label": "no_fallacy"},
    {"text": "I'm going to take a nap.", "true_label": "no_fallacy"},
    {"text": "The keys are on the table.", "true_label": "no_fallacy"},
    {"text": "I'm looking forward to the weekend.", "true_label": "no_fallacy"},
    {"text": "The mountain is covered in snow.", "true_label": "no_fallacy"},
    {"text": "I'm happy to meet you.", "true_label": "no_fallacy"},

    # Slippery Slope (37 examples)
    {"text": "If we let students use calculators, they will forget how to add, and soon nobody will know basic math.", "true_label": "slippery_slope"},
    {"text": "If I give you a cookie, you'll want a glass of milk, and then you'll want a whole meal.", "true_label": "slippery_slope"},
    {"text": "If we allow this small change, soon and the whole system will collapse.", "true_label": "slippery_slope"},
    {"text": "If you don't study now, you'll fail this test, then you'll fail the year, and then you'll never get a job.", "true_label": "slippery_slope"},
    {"text": "If we legalize one drug, soon they will all be legal, and society will fall apart.", "true_label": "slippery_slope"},
    {"text": "If we let the kids stay up late one night, they'll never want to go to bed early again.", "true_label": "slippery_slope"},
    {"text": "If you spend all your money on coffee, you'll go bankrupt and end up on the street.", "true_label": "slippery_slope"},
    {"text": "If we allow people to work from home, soon nobody will ever come to the office again.", "true_label": "slippery_slope"},
    {"text": "If we don't fix this leak, the whole house will eventually flood and be destroyed.", "true_label": "slippery_slope"},
    {"text": "If we let one person break the rules, everyone will break them, and we'll have anarchy.", "true_label": "slippery_slope"},
    {"text": "If you skip one gym session, you'll lose all your progress and become lazy.", "true_label": "slippery_slope"},
    {"text": "If we allow self-driving cars, soon people will lose the ability to drive altogether.", "true_label": "slippery_slope"},
    {"text": "If we increase the minimum wage, businesses will close, and the economy will crash.", "true_label": "slippery_slope"},
    {"text": "If you start eating junk food, you'll become addicted and your health will fail.", "true_label": "slippery_slope"},
    {"text": "If we allow more immigrants, our culture will disappear and we'll lose our identity.", "true_label": "slippery_slope"},
    {"text": "If you tell one lie, you'll become a habitual liar and nobody will ever trust you again.", "true_label": "slippery_slope"},
    {"text": "If we allow AI to write code, soon humans won't be able to understand how software works.", "true_label": "slippery_slope"},
    {"text": "If you don't save for retirement, you'll be destitute and have to work until you're 90.", "true_label": "slippery_slope"},
    {"text": "If we allow animal testing, soon we'll be testing on humans.", "true_label": "slippery_slope"},
    {"text": "If you start smoking, you'll end up with lung cancer and die a painful death.", "true_label": "slippery_slope"},
    {"text": "If we allow video games in school, students will never focus on their lessons again.", "true_label": "slippery_slope"},
    {"text": "If you don't listen to me, you'll make a mistake, and that mistake will ruin your life.", "true_label": "slippery_slope"},
    {"text": "If we allow censorship of one book, soon all books will be banned.", "true_label": "slippery_slope"},
    {"text": "If you ignore this warning, you'll regret it for the rest of your life.", "true_label": "slippery_slope"},
    {"text": "If we allow robots to do our jobs, humans will become redundant and starve.", "true_label": "slippery_slope"},
    {"text": "If you start gambling, you'll lose everything you own and be in debt forever.", "true_label": "slippery_slope"},
    {"text": "If we allow this new building, soon the whole city will be a concrete jungle.", "true_label": "slippery_slope"},
    {"text": "If you don't brush your teeth, they'll all fall out and you'll never eat solid food again.", "true_label": "slippery_slope"},
    {"text": "If we allow more surveillance, we'll end up in a dystopian police state.", "true_label": "slippery_slope"},
    {"text": "If you don't vote, you're responsible for the downfall of democracy.", "true_label": "slippery_slope"},
    {"text": "If we allow plastic bags, the entire ocean will be filled with garbage.", "true_label": "slippery_slope"},
    {"text": "If you start drinking, you'll become an alcoholic and lose your family.", "true_label": "slippery_slope"},
    {"text": "If we allow genetic engineering, we'll create a master race and enslave everyone else.", "true_label": "slippery_slope"},
    {"text": "If you don't pay your taxes, you'll go to prison and your life will be over.", "true_label": "slippery_slope"},
    {"text": "If we allow space travel, we'll bring alien diseases back to Earth and kill everyone.", "true_label": "slippery_slope"},
    {"text": "If you don't wear a seatbelt, you'll die in a crash the first time you drive.", "true_label": "slippery_slope"},
    {"text": "If we allow drones, nobody will ever have privacy again.", "true_label": "slippery_slope"},

    # Strawman (37 examples)
    {"text": "You say we should protect the environment, so you basically want us to live in caves and give up all technology.", "true_label": "strawman"},
    {"text": "My opponent wants to reduce military spending; clearly, he wants our country to be completely defenseless.", "true_label": "strawman"},
    {"text": "You want to increase the minimum wage, so you want all small businesses to go bankrupt.", "true_label": "strawman"},
    {"text": "She wants more gun control, so she wants to take away everyone's right to self-defense.", "true_label": "strawman"},
    {"text": "You think we should spend more on education, so you want to raise taxes until everyone is poor.", "true_label": "strawman"},
    {"text": "He wants to expand public transit, so he wants to ban cars and force us to ride buses forever.", "true_label": "strawman"},
    {"text": "You think we should be more lenient with the dress code, so you want students to come to school in their underwear.", "true_label": "strawman"},
    {"text": "She wants to reform the healthcare system, so she wants government-controlled medicine where nobody gets proper care.", "true_label": "strawman"},
    {"text": "You think we should reduce our carbon footprint, so you want us to stop using electricity and walk everywhere.", "true_label": "strawman"},
    {"text": "He wants to change the zoning laws, so he wants to destroy the character of our neighborhood.", "true_label": "strawman"},
    {"text": "You want to increase immigration, so you want to open the borders to every criminal in the world.", "true_label": "strawman"},
    {"text": "She wants more diversity in the workplace, so she wants to hire people based on quotas rather than merit.", "true_label": "strawman"},
    {"text": "You think we should invest in renewable energy, so you want to shut down every coal power plant tomorrow.", "true_label": "strawman"},
    {"text": "He wants to reform the criminal justice system, so he wants to let all the murderers and thieves out of prison.", "true_label": "strawman"},
    {"text": "You think we should have more affordable housing, so you want the government to build massive ugly skyscrapers everywhere.", "true_label": "strawman"},
    {"text": "She wants to change the school curriculum, so she wants to brainwash our children with radical ideas.", "true_label": "strawman"},
    {"text": "You want to reduce the use of pesticides, so you want us all to starve because the crops will fail.", "true_label": "strawman"},
    {"text": "He wants to increase taxes on the wealthy, so he wants to punish success and destroy the economy.", "true_label": "strawman"},
    {"text": "You think we should have more transparency in government, so you want to leak national secrets to our enemies.", "true_label": "strawman"},
    {"text": "She wants to reform the banking system, so she wants the government to take over all our private accounts.", "true_label": "strawman"},
    {"text": "You think we should have more arts funding, so you want to waste money on paintings that nobody understands.", "true_label": "strawman"},
    {"text": "He wants to increase the number of bike lanes, so he wants to make it impossible for anyone to drive a car.", "true_label": "strawman"},
    {"text": "You want to reduce plastic waste, so you want us to stop consuming anything packaged in plastic.", "true_label": "strawman"},
    {"text": "She wants to reform the election process, so she wants to make it easier for people to commit voter fraud.", "true_label": "strawman"},
    {"text": "You think we should have more animal welfare laws, so you want to ban all meat consumption.", "true_label": "strawman"},
    {"text": "He wants to change the city's budget, so he wants to defund the police and fire departments.", "true_label": "strawman"},
    {"text": "You want to increase the use of public libraries, so you want to shut down all the bookstores.", "true_label": "strawman"},
    {"text": "She wants to reform the tax code, so she wants to create a system where nobody pays their fair share.", "true_label": "strawman"},
    {"text": "You think we should have more community centers, so you want to waste taxpayers' money on things people won't use.", "true_label": "strawman"},
    {"text": "He wants to increase the number of parks, so he wants to stop all new housing development.", "true_label": "strawman"},
    {"text": "You want to reduce the use of cars in the city center, so you want to kill all the local businesses.", "true_label": "strawman"},
    {"text": "She wants to reform the welfare system, so she wants to give money away to people who don't want to work.", "true_label": "strawman"},
    {"text": "You think we should have more school choice, so you want to destroy the public school system.", "true_label": "strawman"},
    {"text": "He wants to change the way we evaluate employees, so he wants to get rid of all accountability.", "true_label": "strawman"},
    {"text": "You want to increase funding for NASA, so you want to waste billions on space while people are suffering on Earth.", "true_label": "strawman"},
    {"text": "She wants to reform the judicial system, so she wants to appoint judges who will ignore the law.", "true_label": "strawman"},
    {"text": "You think we should have more social programs, so you want to turn the country into a communist state.", "true_label": "strawman"}
]

# Ensure exactly 300 (adjusting if necessary)
if len(test_data) > 300:
    test_data = test_data[:300]
elif len(test_data) < 300:
    # Pad with some extra 'no_fallacy' if needed, though the above lists should be close
    while len(test_data) < 300:
        test_data.append({"text": f"This is an extra statement {len(test_data)} for padding.", "true_label": "no_fallacy"})

# Clean up any potential label key errors in the manually written list
for item in test_data:
    if "2rue_label" in item:
        item["true_label"] = item.pop("2rue_label")

print(f"Total test examples: {len(test_data)}")

print("Loading model and tokenizer...")
model_name = "RowdyI7er/DebateLLM"
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="debate_fallacy_model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, subfolder="debate_fallacy_model")

# If CUDA available, use it
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

results = []

print(f"Running inference on {len(test_data)} examples...")
start_time = time.time()

# Process in batches for speed
batch_size = 16
for i in tqdm(range(0, len(test_data), batch_size)):
    batch = test_data[i:i+batch_size]
    texts = [item["text"] for item in batch]
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = logits.argmax(dim=-1).tolist()
    
    for item, pred_id in zip(batch, predictions):
        predicted_label = model.config.id2label[pred_id]
        results.append({
            "text": item["text"],
            "true_label": item["true_label"],
            "predicted_label": predicted_label,
            "is_correct": predicted_label == item["true_label"]
        })

end_time = time.time()

print("Evaluation complete. Saving results...")
with open("evaluation_results_300.json", "w") as f:
    json.dump({
        "results": results, 
        "time_taken": end_time - start_time,
        "batch_size": batch_size,
        "device": device
    }, f, indent=4)
print("Done!")
