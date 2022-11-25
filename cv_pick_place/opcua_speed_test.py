# fmt: off

import time
import random
import asyncio

numWrites = 1

######################################
## ASYNCUA, ONE NODE, WITH REGISTERING
######################################

import asyncua
import asyncua.ua

# Define async function
async def asyncua_one_node_registering():
    # Client instance created like in sync library
    # await keyword is used before most methods
    client = asyncua.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
    await client.connect()
    await client.load_data_type_definitions(overwrite_existing=True)

    # Get struct node and register it for faster access
    Start_Pos = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"')
    Start_Pos = (await client.register_nodes([Start_Pos]))[0]

    # After calling client.load_data_type_definitions(), asyncua.ua module contains classes for all custom types downloaded from server
    pos = asyncua.ua.E6POS()

    # Test writing to the node in a loop
    start_time = time.time()
    for _ in range(numWrites):
        pos.X = round(random.random() * 100, 2)
        pos.Y = round(random.random() * 100, 2)
        pos.Z = round(random.random() * 100, 2)
        pos.A = round(random.random() * 100, 2)
        pos.B = round(random.random() * 100, 2)
        pos.C = round(random.random() * 100, 2)
        pos.Status = 42
        pos.Turn = 2
        await Start_Pos.write_value(asyncua.ua.DataValue(pos))
    end_time = time.time()
    duration = end_time - start_time
    print("ASYNCUA 1 registered node:", duration, "s")

    await client.disconnect()

# Run async function
asyncio.run(asyncua_one_node_registering())

##############################################
## ASYNCUA-SYNC, ONE NODE, WITHOUT REGISTERING
##############################################

import asyncua.sync
import asyncua.ua

client = asyncua.sync.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
client.connect()
client.load_data_type_definitions()

Start_Pos = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"')
pos = asyncua.ua.E6POS()

start_time = time.time()
for _ in range(numWrites):
    pos.X = round(random.random() * 100, 2)
    pos.Y = round(random.random() * 100, 2)
    pos.Z = round(random.random() * 100, 2)
    pos.A = round(random.random() * 100, 2)
    pos.B = round(random.random() * 100, 2)
    pos.C = round(random.random() * 100, 2)
    pos.Status = 42
    pos.Turn = 2
    Start_Pos.write_value(asyncua.ua.DataValue(pos))
end_time = time.time()
duration = end_time - start_time
print("ASYNCUA-SYNC 1 unregistered node:", duration, "s")

client.disconnect()

####################################
## OPCUA, ONE NODE, WITH REGISTERING
####################################

import opcua
import opcua.ua

client = opcua.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
client.connect()
client.load_type_definitions()

Start_Pos = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"')
Start_Pos = client.register_nodes([Start_Pos])[0]

pos = opcua.ua.E6POS()

start_time = time.time()

for _ in range(numWrites):
    pos.X = round(random.random() * 100, 2)
    pos.Y = round(random.random() * 100, 2)
    pos.Z = round(random.random() * 100, 2)
    pos.A = round(random.random() * 100, 2)
    pos.B = round(random.random() * 100, 2)
    pos.C = round(random.random() * 100, 2)
    pos.Status = 42
    pos.Turn = 2
    Start_Pos.set_value(opcua.ua.DataValue(pos))

end_time = time.time()
duration = end_time - start_time
print("OPCUA 1 registered node:", duration, "s")

client.disconnect()

##########################################
## OPCUA, SEPARATE NODES, WITH REGISTERING
##########################################

import opcua
import opcua.ua

client = opcua.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/", 4)
client.connect()

Start_Pos_X = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."X"')
Start_Pos_Y = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Y"')
# Start_Pos_Z = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Z"')
# Start_Pos_A = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."A"')
# Start_Pos_B = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."B"')
# Start_Pos_C = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."C"')
# Start_Pos_Status = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Status"')
# Start_Pos_Turn = client.get_node('ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Turn"')

Start_Pos_X = client.register_nodes([Start_Pos_X])[0]
Start_Pos_Y = client.register_nodes([Start_Pos_Y])[0]
# Start_Pos_Z = client.register_nodes([Start_Pos_Z])[0]
# Start_Pos_A = client.register_nodes([Start_Pos_A])[0]
# Start_Pos_B = client.register_nodes([Start_Pos_B])[0]
# Start_Pos_C = client.register_nodes([Start_Pos_C])[0]
# Start_Pos_Status = client.register_nodes([Start_Pos_Status])[0]
# Start_Pos_Turn = client.register_nodes([Start_Pos_Turn])[0]

nodes = []
nodes.append(Start_Pos_X)
nodes.append(Start_Pos_Y)
# nodes.append(Start_Pos_Z)
# nodes.append(Start_Pos_A)
# nodes.append(Start_Pos_B)
# nodes.append(Start_Pos_C)
# nodes.append(Start_Pos_Status)
# nodes.append(Start_Pos_Turn)

start_time = time.time()

for _ in range(numWrites):
    values = []
    values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(round(random.random() * 100, 2), opcua.ua.VariantType.Float)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(42, opcua.ua.VariantType.Int16)))
    # values.append(opcua.ua.DataValue(opcua.ua.Variant(2, opcua.ua.VariantType.Int16)))
    client.set_values(nodes, values)

end_time = time.time()
duration = end_time - start_time
print("OPCUA 8 registered nodes:", duration, "s")

client.disconnect()
