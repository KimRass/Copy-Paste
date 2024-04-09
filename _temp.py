class COCODS(Dataset):
    def __init__(self, annot_path, img_dir, transform):
        with open(annot_path, mode="r") as f:
            self.annots = json.load(f)

        img_id_to_img_path = dict()
        for img_dict in self.annots["images"]:
            img_id = img_dict["id"]
            img_name = img_dict["file_name"]
            img_path = Path(img_dir)/img_name
            img_id_to_img_path[img_id] = str(img_path)

        self.img_path_to_annots = defaultdict(list)
        for annot in self.annots["annotations"]:
            img_id = annot["image_id"]
            img_path = img_id_to_img_path[img_id]
            self.img_path_to_annots[img_path].append(annot)
        self.img_paths = list(self.img_path_to_annots.keys())

        self.transform = transform

    def __len__(self):
        return len(self.img_path_to_annots)

    @staticmethod
    def annot_to_mask(h, w, annot):
        mask = np.zeros((h, w), dtype=np.uint8)
        for points in annot["segmentation"]:
            # print(np.array(points).shape)
            poly = np.array(points).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, pts=[poly], color=255)
        return mask

    def annots_to_mask(self, h, w, annots):
        masks = list()
        for annot in annots:
            print(annot.keys())
            mask = self.annot_to_mask(h=h, w=w, annot=annot)
            masks.append(mask)
        return torch.from_numpy(np.stack(masks, axis=0))

    def __getitem__(self, idx):
        # idx = 10
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image.show()
        h, w = image.size
        annots = self.img_path_to_annots[img_path]
        # print(img_path)
        # print(annots)
        mask = self.annots_to_mask(h=h, w=w, annots=annots)
        image = self.transform(image)
        # print(image.shape, )
        return image, {"mask": mask}

    def collate_fn(self, batch):
        images = list()
        for image, annot in batch:
            images.append(image)
        return torch.stack(images, dim=0), annot
