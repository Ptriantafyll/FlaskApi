import fasttext
import fasttext.util


ft = fasttext.load_model(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\cc.el.300.bin")
# print(ft.get_dimension())
# fasttext.util.reduce_model(ft, 100)
# print(ft.get_dimension())
ft.get_nearest_neighbors('τρύπα')