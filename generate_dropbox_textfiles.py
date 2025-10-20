import dropbox.sharing
import dropbox,numpy,json
#This script uses the dropbox-sdk-python
#https://github.com/dropbox/dropbox-sdk-python?tab=readme-ov-file
leg_band_colors=["W","Y","R","Special"]
dropbox_folder="D:/Lab Dropbox/Shared_Gerrik/Fake Birds/"
number_of_cards=numpy.linspace(1,300,num=2)
ACCESS_TOKEN ="o3ub1iwp6mu570k"
dbx=dropbox.Dropbox(ACCESS_TOKEN)
data={
    "ID": 
        {"HD":"mm/dd/yy",
        "Father": "John",
        "Mother": "Jane",
        "Children": "N/A",
        "Mates": "N/A"},
    "Manipulations":{"Date":"Info"},
    "Notes":""
    }
json_string = json.dumps(data, indent=4)

for I in leg_band_colors:
    for II in number_of_cards:
        file_address=dropbox_folder+str(II)+I+".txt"
        with open(file_address,'w') as birdfile:
            birdfile.write(json_string)

        with open(file_address, 'rb') as birdfile:
            dbx.files_upload(birdfile.read(), file_address, mode=dropbox.files.WriteMode("overwrite"))
        x=dbx.sharing_create_shared_link_with_settings("Lab Dropbox/Shared_Gerrik/Fake Birds/"+str(II)+I+".txt")
        print(x.url)