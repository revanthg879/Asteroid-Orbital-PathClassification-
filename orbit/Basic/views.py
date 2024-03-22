from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import sklearn
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def Home(request):
    return render(request,'Home.html')

def Index(request):
    return render(request,'Index.html')



def SVM(NEO_flag,One_km_NEO_flag,PHA_flag,H,G,Num_obs,rms,U,Epoch,M,Peri,Node,i,e,n,a,Num_opps,Tp,Orbital_period,Perihelion_dist,Aphelion_dist,Semilatus_rectum,Synodic_period):
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score


    path="C:\\Users\\Dell\\Desktop\\orbit\\orbit\\Basic\\orbital_path.csv"
    data=pd.read_csv(path)

    le_Orbit_type=LabelEncoder()
    data['Orbit_type_n']=le_Orbit_type.fit_transform(data['Orbit_type'])

    inputs =data.drop(['Orbit_type_n','Orbit_type'],'columns')
    output=data['Orbit_type_n']

    x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    model =SVC() #initialzing my model
    model.fit(x_train,y_train)

    y_pred= model.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)*100

    newinputs = np.array([[float(NEO_flag),float(One_km_NEO_flag),float(PHA_flag),float(H),float(G),int(Num_obs),float(rms),int(U),float(Epoch),float(M),float(Peri),float(Node),float(i),float(e),float(n),float(a),int(Num_opps),float(Tp),float(Orbital_period),float(Perihelion_dist),float(Aphelion_dist),float(Semilatus_rectum),float(Synodic_period)]])
    newinputs = sc.transform(newinputs)
    result=model.predict(newinputs)
    return result[0]

def dt(NEO_flag,One_km_NEO_flag,PHA_flag,H,G,Num_obs,rms,U,Epoch,M,Peri,Node,i,e,n,a,Num_opps,Tp,Orbital_period,Perihelion_dist,Aphelion_dist,Semilatus_rectum,Synodic_period):
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    path="C:\\Users\\Dell\\Desktop\\orbit\\orbit\\Basic\\orbital_path.csv"
    data=pd.read_csv(path)

    le_Orbit_type=LabelEncoder()
    data['Orbit_type_n']=le_Orbit_type.fit_transform(data['Orbit_type'])

    inputs =data.drop(['Orbit_type_n','Orbit_type'],'columns')
    output=data['Orbit_type_n']

    x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2)
    model=tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
        
    acc=accuracy_score(y_test, y_pred)*100
    result=model.predict([[float(NEO_flag),float(One_km_NEO_flag),float(PHA_flag),float(H),float(G),int(Num_obs),float(rms),int(U),float(Epoch),float(M),float(Peri),float(Node),float(i),float(e),float(n),float(a),int(Num_opps),float(Tp),float(Orbital_period),float(Perihelion_dist),float(Aphelion_dist),float(Semilatus_rectum),float(Synodic_period)]])
    return result[0]

def Asteroid(request):
    if(request.method=="POST"):
        data=request.POST
        algo=data.get("alg")
        NEO_flag =data.get("txtNEO_flag")
        One_km_NEO_flag =data.get("txtOne_km_NEO_flag")
        PHA_flag =data.get("txtPHA_flag")
        H =data.get("txtH")
        G =data.get("txtG")
        Num_obs =data.get("txtNum_obs")
        rms =data.get("txtrms")
        U =data.get("txtU")
        Epoch =data.get("txtEpoch")
        M =data.get("txtM")
        Peri =data.get("txtPeri")
        Node =data.get("txtNode")
        i =data.get("txti")
        e =data.get("txte")
        n =data.get("txtn")
        a =data.get("txta")
        Num_opps =data.get("txtNum_opps")
        Tp =data.get("txtTp")
        Orbital_period=data.get("txtOrbital_period")
        Perihelion_dist =data.get("txtPerihelion_dist")
        Aphelion_dist =data.get("txtAphelion_dist")
        Semilatus_rectum =data.get("txtSemilatus_rectum")
        Synodic_period =data.get("txtSynodic_period")
        if algo==0:
            result=SVM(NEO_flag,One_km_NEO_flag,PHA_flag,H,G,Num_obs,rms,U,Epoch,M,Peri,Node,i,e,n,a,Num_opps,Tp,Orbital_period,Perihelion_dist,Aphelion_dist,Semilatus_rectum,Synodic_period)
        else:
            result=dt(NEO_flag,One_km_NEO_flag,PHA_flag,H,G,Num_obs,rms,U,Epoch,M,Peri,Node,i,e,n,a,Num_opps,Tp,Orbital_period,Perihelion_dist,Aphelion_dist,Semilatus_rectum,Synodic_period)
        if (result==0):
            prediction="AMOR"
        elif(result==1):
            prediction="APOLLO"
        elif(result==2):
            prediction="ATENAA"
        elif(result==3):
            prediction="ATIRA"
        elif(result==4):
            prediction="DISTANT OBJECT"
        elif(result==5):
            prediction="HILDA"
        elif(result==6):
            prediction="HUNGARIA"
        elif(result==7):
            prediction="JUPITER TROJAN"
        elif(result==8):
            prediction="MBA"
        elif(result==9):
            prediction="OBJECT WITH PERIHELION <1.665 AU"
        else:
            prediction="PHOCAEA"
        
        return render(request,"Asteroid.html",context={'prediction':prediction})

    return render(request,"Asteroid.html")






def Dt(PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched):
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    path="C:\\Users\\Dell\\Desktop\\orbit\\orbit\\Basic\\network_intrusion.csv"
    data=pd.read_csv(path)
        
    inputs =data.drop(['Label','Packets Rx Dropped','Packets Tx Dropped','Packets Rx Errors','Packets Tx Errors','Delta Packets Rx Dropped','Delta Packets Tx Dropped','Delta Packets Rx Errors','Delta Packets Tx Errors','is_valid','Table ID','Max Size'],'columns')
    output=data['Label']
        
    x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2)
    model=tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
        
    acc=accuracy_score(y_test, y_pred)*100
    result=model.predict([[PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched]])
    return result[0]

def svm(PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched):
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    path="C:\\Users\\Dell\\Desktop\\orbit\\orbit\\Basic\\network_intrusion.csv"
    data=pd.read_csv(path)
    inputs =data.drop(['Label','Packets Rx Dropped','Packets Tx Dropped','Packets Rx Errors','Packets Tx Errors','Delta Packets Rx Dropped','Delta Packets Tx Dropped','Delta Packets Rx Errors','Delta Packets Tx Errors','is_valid','Table ID','Max Size'],'columns')
    output=data['Label']
        
    x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    model =SVC() #initialzing my model
    model.fit(x_train,y_train)

    y_pred= model.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)*100
    result=model.predict([[PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched]])
    return result[0]

def Network(request):
    if(request.method=="POST"):
        data=request.POST
        algo=data.get("radio")
        PortNumber=data.get("txtPort Number")
        ReceivedPackets=data.get("txtReceived Packets")
        ReceivedBytes=data.get("txtReceived Bytes")
        SentBytes=data.get("txtSent Bytes")
        SentPackets=data.get("txtSent Packets")
        PortaliveDuration=data.get("txtPort alive Duration")
        DeltaReceivedPackets=data.get("txtDelta Received Packets")
        DeltaReceivedBytes=data.get("txtDelta Received Bytes")
        DeltaSentBytes=data.get("txtDelta Sent Bytes")
        DeltaSentPackets=data.get("txtDelta Sent Packets")
        DeltaPortaliveDuration=data.get("txtDelta Port alive Duration")
        ConnectionPoint=data.get("txtConnection Point")
        TotalLoadRate=data.get("txtTotal Load/Rate")
        TotalLoadLatest= data.get("txtTotal Load/Latest")
        UnknownLoadRate=data.get("txtUnknown Load/Rate")
        UnknownLoadLatest=data.get("txtUnknown Load/Latest")
        Latestbytescounter=data.get("txtLatest bytes counter")
        ActiveFlowEntries=data.get("txtActive Flow Entries")
        PacketsLookedUp=data.get("txtPackets Looked Up")
        PacketsMatched=data.get("txtPackets Matched")
        if algo==1:
            result=Dt(PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched)
        elif algo==0:
            result=svm(PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched)
        else:
            result=Dt(PortNumber,ReceivedPackets,ReceivedBytes,SentBytes,SentPackets,PortaliveDuration,DeltaReceivedPackets,DeltaReceivedBytes,DeltaSentBytes,DeltaSentPackets,DeltaPortaliveDuration,ConnectionPoint,TotalLoadRate,TotalLoadLatest,UnknownLoadRate,UnknownLoadLatest,Latestbytescounter,ActiveFlowEntries,PacketsLookedUp,PacketsMatched)

        if(result==0):
            prediction="Normal"
        elif(result==1):
            prediction="BlackHole"
        elif(result==2):
            prediction="TCP-SYN"
        elif(result==3):
            prediction="PortScan"
        elif(result==4):
            prediction="Diversion"
        else:
            prediction="Overflow"
        return render(request,"Network.html",context={'prediction':prediction})

    return render(request,"Network.html")


