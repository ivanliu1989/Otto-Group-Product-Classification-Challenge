import datetime
from datetime import datetime
import numpy as np

def get_label(tipo):
 labels={'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4,'Class_5':5,'Class_6':6,'Class_7':7,'Class_8':8,'Class_9':9}
 label=labels[tipo]
 return label
def csv_to_vw(loc_csv, loc_output, train=True):
    start = datetime.now()
    col_train=['id','target']
    col_test=['id']
    for k in range(1,94):
        col_train.append(str(k))
        col_test.append(str(k))
    
    print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
    i = open(loc_csv, "r")
    j = open(loc_output, 'wb')
    counter=0
    with i as infile:
        line_count=0
        for line in infile:
            # to counter the header
            if line_count==0:
                line_count=1
                continue
        
            categorical_features = ""
            counter = counter+1
    
            line = line.split(",")
            if train:
                col=col_train
                for i in range(2,95):
                    if line[1] != "":
                        categorical_features += "|feat_%s %s" % (col[i],line[i-1])
                categorical_features+='\n'        
            else:
                col=col_test
                for i in range(1,94):
                    if line[i] != "":
                        categorical_features += "|feat_%s %s" % (col[i],line[i])
               
             
            if train: #we care about labels
                
                label=get_label(line[94].rstrip('\n'))
               
                j.write( "%s '%s %s" % (label,line[0],categorical_features))

            else: #we dont care about labels
                j.write("1 '%s %s" % (line[0],categorical_features) )

  #Reporting progress
            #print counter
            if counter % 1000== 0:
                print("%s\t%s"%(counter, str(datetime.now() - start)))

    print("\n %s Task execution time:\n\t%s"%(counter, str(datetime.now() - start)))

csv_to_vw("path_to_yourdata/train.csv", "./train.vw",train=True)
csv_to_vw("path_to_yourdata/test.csv", "./test.vw",train=False)
