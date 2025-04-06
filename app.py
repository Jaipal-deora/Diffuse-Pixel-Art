import streamlit as st 
import torch 
from src.helper import sample_ddpm_context, show_images, sample_ddim_context
from src.unet import ContextUNET

SAMPLE_TYPES = ['DDPM', 'DDIM']
CONTEXT_LABELS = ['HERO', 'NON-HERO', 'FOOD', 'SPELL', 'SIDE-FACING']
device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))
save_dir = 'weights'

nn_model = ContextUNET(in_channels=3, n_feat=64, n_cfeat=5, height=16).to(device)
nn_model.load_state_dict(torch.load(f"{save_dir}/model.pth", map_location=device,weights_only=True))
nn_model.eval()
print("Loaded in Model")

def main():
    st.title('Pixel Art Diffusion')
    st.markdown('Set context, # of samples, and sampling method to generate image.')
    

    with st.sidebar:
        num_samples = st.slider("Number of samples", 1, 32, 4)
        sample_type = st.selectbox("Sampling method", SAMPLE_TYPES)
        context = [st.checkbox(label) for label in CONTEXT_LABELS]
        generate_btn = st.button("Create Images")

    if generate_btn:
        with st.spinner('Generating images ...'):
            context_onehot = torch.tensor([int(c) for c in context]).unsqueeze(0).repeat(num_samples,1).float().to(device)
            # print(context_onehot)
            if sample_type == 'DDPM':
                samples, _ = sample_ddpm_context(nn_model,context_onehot.shape[0], context_onehot)
            else:
                samples, _ = sample_ddim_context(nn_model,context_onehot.shape[0],context_onehot)
            fig = show_images(samples)
            st.pyplot(fig)

main()