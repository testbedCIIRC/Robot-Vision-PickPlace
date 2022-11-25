import opcua
import opcua.ua
import time

# client = opcua.Client("opc.tcp://OpcUaOperator:kuka@10.35.91.6:4840/", 4)
# client.connect()
# client.load_type_definitions()

# # Start_Pos = client.get_node('ns=5;s="MotionDeviceSystem"."ProcessData"."R1"."Program"."opcua_variables"."Xtestvar"')
# Start_Pos = client.get_node('ns=5;s=MotionDeviceSystem.ProcessData.R1.Program.opcua_variables.Xtestvar')

# pos = opcua.ua.E6Pos()

# pos.X = 1.0
# pos.Y = 2.0
# pos.Z = 3.0
# pos.A = 4.0
# pos.B = 5.0
# pos.C = 6.0

# start_time = time.time()

# print(Start_Pos.get_value())
# Start_Pos.set_value(opcua.ua.DataValue(pos))
# print(Start_Pos.get_value())

# end_time = time.time()
# duration = end_time - start_time
# print("Time:", duration, "s")

# client.disconnect()

client = opcua.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
client.connect()
client.load_type_definitions()

# Start_Pos = client.get_node('ns=5;s="MotionDeviceSystem"."ProcessData"."R1"."Program"."opcua_variables"."Xtestvar"')
Start_Pos = client.get_node('ns=3;s="Robot_Data"."matrix"')

m = Start_Pos.get_value()
print(type(m))

m[0][0] = 13.0

start_time = time.time()
Start_Pos.set_value(opcua.ua.DataValue(m))
end_time = time.time()
duration = end_time - start_time
print("Time:", duration, "s")

client.disconnect()
