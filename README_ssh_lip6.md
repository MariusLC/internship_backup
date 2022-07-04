se connecter en ssh :

avec putty :
ouvrir putty.exe (dans C:Programmes/PuTTy/)

ou en lignes de commandes :
ssh -L 2222:fosterthepeople:22 gate.lip6.fr (a verif)


Commande générale upload/download :
scp (-r pour un dossier entier) My/local/Path/Source/FILENAME lechapelier@gate.lip6.fr:My/Remote/Path/Destination/

Uploader un fichier sur une machine du labo (depuis anacondaprompt):
scp C:\Users\mariu\Desktop\Travail\FILENAME lechapelier@gate.lip6.fr:

Downloader un fichier depuis une machine du labo (depuis anacondaprompt):
scp lechapelier@gate.lip6.fr:FILENAME C:\Users\mariu\Desktop\Travail