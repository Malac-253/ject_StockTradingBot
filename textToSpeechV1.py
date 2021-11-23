from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('0QTcC9UVs9s_zCYPf4Zlpbg-TOUK6OrbVsUuWxDNiCF6')
text_to_speech = TextToSpeechV1(
    authenticator=authenticator
)

text_to_speech.set_service_url('https://api.us-east.text-to-speech.watson.cloud.ibm.com')



voices = text_to_speech.list_voices().get_result()
print(json.dumps(voices, indent=2))


with open('hello_world.wav', 'wb') as audio_file:
    audio_file.write(
        text_to_speech.synthesize(
            'Hello , Captain Beerram',
            voice='en-US_AllisonV3Voice',
            accept='audio/wav'        
        ).get_result().content)