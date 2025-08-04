from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the tokenizer and model
MODEL_NAME = "facebook/bart-large-cnn"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Your input text
input_text = """
Senior Advocate Dr Abhishek Manu Singhvi, for Rahul Gandhi, at the outset submitted that if an opposition leader cannot raise issues, it would be an unfortunate situation. "If he can't say these things which are published in the Press, he can't be a leader of opposition," Singhvi said. "Whatever you have to say, why don't you say in the Parliament? Why do you have to say this in the social media posts?" Justice Datta asked. Also Read - Tamil Nadu Govt Moves Supreme Court Against HC Restraint On Naming Welfare Schemes After Living Persons, Former CMs Expressing further disapproval of Gandhi's comments, Justice Datta asked, "Tell Dr.Singhvi, how do you get to know that 2000 square kilometres of Indian territory were occupied by the Chinese? Were you there? Do you have any credible material? Why do you make these statements without any...If you were a true Indian, you would not say all this." "It is also possible that a true Indian will say that our 20 Indian soldiers were beaten up and killed and that it is a matter of concern," Singhvi replied. Also Read - Supreme Court Rebukes Bihar IPS Officer For Filing Affidavit Supporting Murder Convict; Issues Show Cause Notice "When there is a conflict across the order, is it unusual to have casualties on both sides?" Justice Datta asked in return. Singhvi said that Gandhi was only on the point of proper disclosure and raising concerns about the suppression of information. But Justice Datta said that there was a proper forum to raise the questions. Singhvi, while conceding that the petitioner could have worded the comments in a better manner, said that the complaint was nothing but an attempt to harass him only for raising questions, which is the duty of an opposition leader. He also pointed out that as per Section 223 BNSS, prior hearing of the accused was mandatory before taking cognisance of a criminal complaint, which has not been complied with in this case. Justice Datta however pointed out that this point of Section 223 was not raised before the High Court. Also Read - Medical Jurisprudence & Toxicology Foundation Course by LiveLaw Academy Singhvi conceded that there was a lapse in raising this point. He said that the challenge in the High Court primarily focused on the locus of the complainant. Singhvi questioned the High Court's reasoning that the complainant, though not a "person aggrieved", is a "person defamed". The bench ultimately agreed to consider this point and issued notice on Gandhi's Special Leave Petition challenging the Allahabad High Court's judgment which refused to quash the proceedings. Interim stay has been granted for a period of three weeks. Senior Advocate Gaurav Bhatia appeared for the complainant on caveat. On May 29, the Allahabad High Court rejected Gandhi's plea, who had moved the HC challenging the defamation case as well as the summoning order passed in February 2025 by an MP MLA court in Lucknow. Justice Subhash Vidyarthi of the High Court observed that freedom of speech and expression does not include the freedom to make statements which are defamatory to the Indian Army. The defamation complaint, filed by former Border Roads Organisation (BRO) Director Uday Shankar Srivastava and presently pending in a court in Lucknow, states that the alleged derogatory remarks by Gandhi were made on December 16, 2022, during his Bharat Jodo Yatra. The complaint adds that Gandhi's objectionable comments, pertaining to a clash between the Indian and Chinese armies on December 9, 2022, had defamed the Indian Army. It has been specifically alleged that Gandhi repetitively stated in a very derogatory manner that the Chinese army is 'thrashing' our soldiers in Arunachal Pradesh, and that the Indian Press will not ask any question in this regard. Challenging the order of the Lucknow Court, wherein it was prima facie observed that Gandhi's statement appeared to have resulted in demoralising the Indian Army and persons attached to it and their family members, Gandhi had moved the High Court.

"""

# Tokenize the input
inputs = tokenizer.encode(
    input_text,
    return_tensors="pt",      # return PyTorch tensors
    max_length=1024,
    truncation=True
)

# Generate summary
summary_ids = model.generate(
    inputs,
    max_length=130,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode summary back to text
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n=== SUMMARY ===\n")
print(summary)
