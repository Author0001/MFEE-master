with open('data_forDis/test.txt','r',encoding='utf-8') as f:
    data = f.read().replace("reason","Reason").replace("earthquake","Earthquake").replace("geofracture","Geofracture").replace("landslide","Landslide").replace("sinking","Sinking").replace("collapse","Collapse").replace("debrisFlow","DebrisFlow").replace("subsidence","Subsidence").replace("movement","Movement").replace("investigate","Investigate").replace("action","Action").replace("Object","Damage").replace("achievement","Achievement")
    with open('data_forDis/test2.txt','w',encoding='utf-8') as wf:
        wf.write(data)