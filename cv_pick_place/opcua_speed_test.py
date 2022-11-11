from asyncua import sync

client = sync.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
client.aio_obj.secure_channel_timeout = 300000
client.aio_obj.session_timeout = 30000
client.connect()

Start_Pos_X = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"')
E6POS_Node = client.get_node('ns=3;s=DT_"E6POS"')

val = Start_Pos_X.get_value()
print(val)

node_types = client.load_data_type_definitions()
pos = node_types["E6POS"]
pos.Status = 42
pos.X = 15

# for t in node_types:
#     print(t)




client.disconnect()
exit()
