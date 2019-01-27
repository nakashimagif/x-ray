def move_trashbox(prop, keep):
    for tt in dir_tt:
        dst_dir = os.path.join(trashbox_dir, tt)
        src_dir = os.path.join(origin_dir, tt)
        os.makedirs(dst_dir, exist_ok=True)
        src_list = os.listdir(src_dir)
        temp_size = int(len(src_list) * prop)
        temp_list = random.sample(src_list, temp_size)
        if keep == True:
            target_list = [
                target for target in src_list if target not in temp_list
            ]
        if keep == False:
            target_list = temp_list
        for target in target_list:
            sfile = os.path.join(src_dir, target)
            shutil.move(sfile, dst_dir)


def restore_trashbox(prop):
    for tt in dir_tt:
        dst_dir = os.path.join(origin_dir, tt)
        src_dir = os.path.join(trashbox_dir, tt)
        src_list = os.listdir(src_dir)
        temp_size = int(len(src_list) * prop)
        target_list = random.sample(src_list, temp_size)
        for target in target_list:
            sfile = os.path.join(src_dir, target)
            shutil.move(sfile, dst_dir)


def pool_image():
    for tt in dir_tt:
        src_dir = os.path.join(origin_dir, tt)
        target_list = os.listdir(src_dir)
        for target in target_list:
            sfile = os.path.join(src_dir, target)
            shutil.move(sfile, origin_dir)