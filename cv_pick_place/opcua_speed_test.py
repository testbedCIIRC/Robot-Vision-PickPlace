import opcua
from opcua.ua.uatypes import DataValue, Variant, VariantType

opcua.ua.uatypes.DataValue

client = opcua.Client("opc.tcp://user:CIIRC@10.35.91.101:4840/")
client.connect()

Home_Pos_X = client.get_node('ns=3;s="Robot_Positions"."Home"."X"')

val = Home_Pos_X.get_value()
print(val)
val = Home_Pos_X.set_value(DataValue(Variant(50, VariantType.Float)))




client.disconnect()
