
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

WATCH_FOLDER = "D:/Pasta Drive/Raíssa/SOJA V2 20032025/SOJA-20-03/AMOSTRAS/Unicos"
BACKGROUND_FOLDER1 = "D:/Pasta Drive/Raíssa/SOJA V2 20032025/SOJA-20-03/BACKGROUND/background1"
BACKGROUND_FOLDER2 = "D:/Pasta Drive/Raíssa/SOJA V2 20032025/SOJA-20-03/BACKGROUND/background1"

MODEL_PATH = "D:/Pasta Drive/Raíssa/SOJA V2 20032025/SOJA-20-03/xgb_model.pkl"
ENCODER_PATH = "D:/Pasta Drive/Raíssa/SOJA V2 20032025/SOJA-20-03/xgb_encoder.pkl"

def carregar_imagem(path, log):
    try:
        log(f"   ➕ Tentando abrir: {path}")
        img = Image.open(path)
        img = np.array(img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        log(f"   ✅ Imagem carregada: shape={img.shape}, dtype={img.dtype}")
        return img
    except Exception as e:
        log(f"   ❌ Erro ao abrir {path}: {e}")
        return None

def load_images_to_cube(folder_path, log):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff', '.bmp', '.png', '.jpeg'))])
    images = []
    for file in files:
        img_path = os.path.join(folder_path, file)
        img = carregar_imagem(img_path, log)
        if img is not None:
            images.append(img)
    if not images:
        raise ValueError(f"Nenhuma imagem válida em: {folder_path}")
    cube = np.stack(images, axis=-1)
    return cube

def load_images_from_subfolders(root_folder, log):
    cubes = {}
    folder_name = os.path.basename(root_folder.rstrip("///"))
    direct_files = [f for f in os.listdir(root_folder) if f.lower().endswith(('.tif', '.tiff', '.bmp', '.png', '.jpeg'))]
    if direct_files:
        log(f"🔍 Encontrado {len(direct_files)} arquivos diretamente em {root_folder}")
        images = []
        for file in sorted(direct_files):
            img_path = os.path.join(root_folder, file)
            img = carregar_imagem(img_path, log)
            if img is not None:
                images.append(img)
        if images:
            cube = np.stack(images, axis=-1)
            cubes[folder_name] = cube
            log(f"   ✅ Cubo montado direto em '{folder_name}': {cube.shape}")
        else:
            log(f"   ❌ Nenhuma imagem válida em {root_folder}")
    return cubes

def extrair_espectros_por_objeto(cube, mask):
    espectros = []
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        object_mask = labeled_mask == region.label
        pixels = cube[object_mask]
        espectro_medio = np.mean(pixels, axis=0)
        espectros.append(espectro_medio)
    return espectros

def classificar_objetos(espectros, model, encoder, log, mask):
    if not espectros:
        log("⚠️ Nenhum espectro para classificar.")
        return None  # Agora retorna None

    predicoes = model.predict(espectros)
    classes = encoder.inverse_transform(predicoes)
    log("🧠 Classificação dos objetos:")
    for i, classe in enumerate(classes, 1):
        log(f" - Objeto {i}: {classe}")
    from collections import Counter
    contagem = Counter(classes)
    for classe, qtd in contagem.items():
        log(f"   🧾 {classe}: {qtd} objeto(s)")
    log(f"   🧪 {len(classes)} classes previstas / {len(np.unique(mask)) - 1} objetos detectados")

    # Gera a imagem aqui
    from skimage.measure import label
    labeled_mask = label(mask)
    imagem_resultado = gerar_imagem_classes(classes, labeled_mask)

    return imagem_resultado


def gerar_imagem_classes(classes, mask):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from collections import Counter

    cor_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0), (0, 128, 0),
        (100, 100, 255), (255, 100, 100), (150, 255, 150),
        (200, 200, 0), (0, 200, 200)
    ]

    h, w = mask.shape
    img_colorida = np.zeros((h, w, 3), dtype=np.uint8)

    # Detectar pixels de fundo com base nos vértices
    bordas = [
        mask[0, 0],           # canto superior esquerdo
        mask[0, -1],          # canto superior direito
        mask[-1, 0],          # canto inferior esquerdo
        mask[-1, -1]          # canto inferior direito
    ]
    valor_fundo = max(set(bordas), key=bordas.count)

    # Zerar todos os pixels que têm o mesmo valor que os vértices
    mask = np.where(mask == valor_fundo, 0, mask)

    unique_labels = np.unique(mask)
    if len(classes) != len(unique_labels) - 1:
        return None

    contagem = Counter(classes)
    cores = {classe: cor_palette[i % len(cor_palette)] for i, classe in enumerate(contagem.keys())}

    for i, label_id in enumerate(unique_labels):
        if label_id == 0:
            continue
        classe = classes[i - 1]
        img_colorida[mask == label_id] = cores[classe]

    imagem_base = Image.fromarray(img_colorida)
    legenda_largura = 200
    nova_img = Image.new("RGB", (w + legenda_largura, h), (255, 255, 255))
    nova_img.paste(imagem_base, (0, 0))

    draw = ImageDraw.Draw(nova_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    y_offset = 10
    for classe, qtd in contagem.items():
        cor = cores[classe]
        draw.rectangle([w + 10, y_offset, w + 30, y_offset + 20], fill=cor)
        draw.text((w + 35, y_offset), f"{classe}: {qtd}", fill=(0, 0, 0), font=font)
        y_offset += 30

    return nova_img

def process_pipeline(folder_path, log, app_instance):

    try:
        bg1 = load_images_to_cube(BACKGROUND_FOLDER1, log)
        bg2 = load_images_to_cube(BACKGROUND_FOLDER2, log)
        log("✔️ Backgrounds carregados com sucesso")
    except Exception as e:
        log(f"❌ Erro ao carregar backgrounds: {e}")
        return

    try:
        cubes = load_images_from_subfolders(folder_path, log)
        if not cubes:
            log("⚠️ Nenhum cubo foi carregado. Verifique o conteúdo da pasta.")
            return

        background_median = np.median(np.stack([bg1, bg2], axis=-1), axis=-1)
        for class_name, cube in cubes.items():
            if cube.shape != background_median.shape:
                log(f"⚠️ Dimensões incompatíveis entre '{class_name}' e background")
                continue
            cube = cube.astype(np.float32)
            corrected = cube - background_median
            summed = np.sum(corrected, axis=-1)
            plt.imshow(summed, cmap='gray')
            plt.title(f"Soma espectral - {class_name}")
            plt.axis('off')
            plt.show()

            from sklearn.cluster import KMeans
            H, W, Z = corrected.shape
            reshaped = corrected.reshape(-1, Z)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reshaped)
            clustered = labels.reshape(H, W)

            mean_values = [np.mean(corrected[clustered == i]) for i in range(2)]
            background_label = int(np.argmin(mean_values))
            mask = (clustered != background_label).astype(np.uint8)

            plt.imshow(mask, cmap='gray')
            plt.title(f"Clusters - {class_name}")
            plt.axis('off')
            plt.show()

            # Mascara limpa (fundo = 0, ignora cluster 2)
            clustered[clustered == 2] = 0

            # Extrair espectros
            from skimage.measure import label
            labeled_mask = label(mask)
            espectros = extrair_espectros_por_objeto(corrected, labeled_mask)


            # Carregar modelo e classificar
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                with open(ENCODER_PATH, 'rb') as f:
                    encoder = pickle.load(f)
                imagem_resultado = classificar_objetos(espectros, model, encoder, log, labeled_mask)
                app_instance.imagem_para_mostrar = imagem_resultado if imagem_resultado else None
            else:
                log("❌ Modelo ou codificador não encontrados.")
                imagem_resultado = classificar_objetos(espectros, model, encoder, log, labeled_mask)
                #app_instance.imagem_para_mostrar = imagem_resultado if imagem_resultado else None


            app_instance.imagem_para_mostrar = imagem_resultado if imagem_resultado else None

            if imagem_resultado:
                app_instance.imagem_para_mostrar = imagem_resultado
            else:
                app_instance.imagem_para_mostrar = None
            log("✅ Processamento completo!")
    except Exception as e:
        log(f"❌ Erro no pipeline: {e}")

class PastaHandler(FileSystemEventHandler):
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def on_created(self, event):
        if event.is_directory:
            msg = f"📁 Nova pasta detectada: {event.src_path}"
            self.log_callback(msg)
            process_pipeline(event.src_path, self.log_callback, self.log_callback.__self__)
            if hasattr(self.log_callback.__self__, 'imagem_para_mostrar'):
                imagem = self.log_callback.__self__.imagem_para_mostrar
                if imagem:
                    self.log_callback.__self__.mostrar_imagem(imagem)


class App:

    def mostrar_imagem(self, imagem):
        if imagem:
            top = tk.Toplevel(self.master)
            top.title("Resultado da Classificação")
            imagem.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(imagem)
            label = tk.Label(top, image=img_tk)
            label.image = img_tk  # mantém referência viva
            label.pack()


    def __init__(self, master):
        self.master = master
        master.title("Classificador Espectral")

        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=70, height=25)
        self.text_area.pack(padx=10, pady=10)
        self.text_area.insert(tk.END, f"Aguardando novas pastas em: {WATCH_FOLDER}")

        self.start_monitoramento()

    def log(self, msg):
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.see(tk.END)

    def start_monitoramento(self):
        if not os.path.exists(WATCH_FOLDER):
            os.makedirs(WATCH_FOLDER)

        event_handler = PastaHandler(self.log)
        observer = Observer()
        observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
        observer_thread = threading.Thread(target=observer.start)
        observer_thread.daemon = True
        observer_thread.start()

        self.log("🚀 Monitoramento iniciado...")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
