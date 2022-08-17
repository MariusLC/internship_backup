se connecter à une machine du lip6 à partir d'une machine exterieure :

	ssh lechapelier@gates.lip6.fr

se connecter au cluster à partir d'une machine du lip6 :

	ssh lechapelier@cluster.lip6.fr


accéder au dépot depuis ma session :

	cd /net/big/lechapelier


lancer une tâche :
	
	oarsub -l "/nodes=1/core=12,walltime=100:00:00" "python3 -m moral.moral_train"
	-> retourne le JOB_ID, ex : JOB_ID=982376


se connecter directement au noeud, puis lancer une tâche :
	
	oarsub -I
	python3 test.py
	exit


annuler une tâche :
	
	oardel [JOB_ID]


visualiser les fichiers de logs/errors :

	cat OAR.[JOB_ID].stderr
	cat OAR.[JOB_ID].stdout

	ou 

	nano OAR.[JOB_ID].stderr
	-> nano permet également de modifier des fichiers à partir du terminal

copier des fichiers :

du pc au cluster :
	cp -r ~/Documents/backup/moral_rl/generated_data/v3/rand/ generated_data/v3/
	cp -r ~/Documents/backup/moral_rl/generated_data/v3/pref_model/ generated_data/v3/

	cp "/net/big/lechapelier/internship_backup/moral_rl/generated_data/v3/moral_agents/DEMOS_[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3_after_queries_fixed.pk" "generated_data/v3/moral_agents/" 

du cluster au pc :
	cp -r generated_data/v1/rand/ ~/Documents/backup/moral_rl/generated_data/v1/
	cp 'generated_data/v3/[0, 0, 1, 0]/expert.pt' '~/Documents/backup/moral_rl/generated_data/v3/[0, 0, 1, 0]/'
	cp -r 'generated_data/v3/[0, 1, 1, 1]' ~/Documents/backup/moral_rl/generated_data/v3/
	cp -r generated_data/v3/moral_agents/ ~/Documents/backup/moral_rl/generated_data/v3/

	cp "generated_data/v3/moral_agents/DEMOS_[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3_after_queries_fixed.pk" "~/Documents/backup/moral_rl/generated_data/v3/moral_agents/"
