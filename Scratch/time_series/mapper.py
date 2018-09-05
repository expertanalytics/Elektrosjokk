import numpy as np

with open("channel.pos", "r") as ifh:
    channels = list(map(lambda x: x.split(), ifh.readlines()))

# with open("channel_old.pos", "r") as ifh:
#     channels = list(map(lambda x: x.split(","), ifh.readlines()))

channel_dict = {
    ch[0]: np.fromiter(map(lambda x: 10*float(x), ch[1:]), dtype=np.float64) for ch in channels
}

for _, v in channel_dict.items():
    print(v)

# assert False
# print(channel_dict.keys())

# with open("register.dat", "r") as ifh:
#     mylines = ifh.readlines()[4:-1]
#     foo = list(map(lambda x: list(map(float, x.strip().split())), mylines))
#     reg_aff = np.asarray(foo)


# print(channel_dict.keys())
# import nibabel as nib
# print(reg_aff)
# print()

# np.set_printoptions(precision=2)

# old_x = np.array(list(channel_dict.values()))


# new_x = nib.affines.apply_affine(reg_aff, old_x)
# for e in new_x:
#     print(list(map(float, e)))
# print(new_x)
