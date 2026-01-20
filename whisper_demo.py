import os
import sys
import warnings
from faster_whisper import WhisperModel

# python whisper_demo.py data/audio/1月20日_1415_会议-5d75ac50.mp3
# 忽略无关警告（可选）
warnings.filterwarnings("ignore")

class OnlineAudioTranscriber:
    """
    纯在线模式的 faster-whisper 语音识别器（自动从 Hugging Face 下载模型）。
    """
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        """
        初始化在线 Whisper 模型（自动下载）。
        
        :param model_size: 模型大小 (tiny/base/small/medium/large-v2/large-v3)
        :param device: 运行设备 (cpu/cuda)
        :param compute_type: 计算精度 (cpu推荐int8/float32，gpu推荐float16)
        """
        # 1. 配置国内访问 Hugging Face 加速（核心：解决超时问题）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 清华镜像
        # 如需代理，取消下面注释并替换为你的代理地址（比如 Clash/梯子）
        # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
        # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        
        # 模型名 -> Hugging Face 官方库映射
        model_alias = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base",
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",  # 你之前用的medium模型
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3"
        }
        self.model_name = model_alias.get(model_size.lower(), model_size)
        
        try:
            print(f"📥 开始加载在线模型: {self.model_name}（首次运行会自动下载，耐心等待）")
            # 初始化模型（在线模式核心：local_files_only=False，允许联网下载）
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                local_files_only=False,  # 强制在线下载/校验模型
                download_root=os.path.expanduser("~/.cache/huggingface/hub")  # 模型缓存路径
            )
            print(f"✅ 在线模型 {model_size} 加载成功！")
        except TimeoutError:
            raise RuntimeError(
                "❌ 模型下载超时！\n"
                "解决方案：\n"
                "1. 检查网络是否能访问 huggingface.co\n"
                "2. 配置代理（取消代码中 HTTP_PROXY/HTTPS_PROXY 的注释）\n"
                "3. 确认 HF_ENDPOINT 镜像配置正确"
            )
        except Exception as e:
            raise RuntimeError(f"❌ 在线模型初始化失败: {str(e)}\n💡 请检查网络/代理配置，或切换更小的模型（如base）测试")

    def transcribe(self, audio_path, verbose=True):
        """
        在线转录音频文件（支持 mp3/wav/m4a/flac 等格式）。
        
        :param audio_path: 音频文件路径
        :param verbose: 是否打印详细日志
        :return: 转录后的文本
        """
        # 1. 校验音频文件
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ 音频文件不存在: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"❌ 音频文件为空: {audio_path}")
        
        # 2. 执行转录
        if verbose:
            print(f"\n📼 开始转录音频: {audio_path}")
            print(f"🔍 音频文件大小: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
        
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language="zh",  # 强制中文识别
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            word_timestamps=True
        )
        
        # 3. 拼接结果
        transcript_parts = [seg.text.strip() for seg in segments]
        full_text = "".join(transcript_parts)
        
        # 4. 打印日志
        if verbose:
            print(f"🌐 检测到语言: {info.language} (置信度: {info.language_probability:.2f})")
            print(f"⏱️  音频时长: {info.duration:.2f} 秒")
            print(f"📝 转录完成，文本长度: {len(full_text)} 字符")
            print("\n=================== 转录结果 ===================")
            print(full_text if len(full_text) <= 500 else full_text[:500] + "...（内容过长，仅展示前500字符）")
            print("===============================================\n")
        
        return full_text

def main():
    """
    在线模式测试主函数：支持命令行传参，格式：python whisper_online_test.py <模型大小> <音频文件路径>
    """
    # 1. 解析命令行参数
    if len(sys.argv) != 3:
        print("📚 在线模式使用说明：")
        print("   方式1（命令行传参）：")
        print("     python whisper_online_test.py <模型大小> <音频文件路径>")
        print("     示例：python whisper_online_test.py medium /data/test_audio.mp3")
        print("\n   模型大小可选：tiny(最快)/base/small/medium(平衡)/large-v3(最准)")
        print("="*60)
        
        # 手动指定（不想用命令行则修改这里）
        MODEL_SIZE = "tiny"  # 可选：tiny/base/small/medium/large-v3
        AUDIO_PATH = "data/audio/1月20日_1415_会议-5d75ac50.mp3"  # 替换为你的音频路径
    else:
        # 命令行传参
        MODEL_SIZE = sys.argv[1]
        AUDIO_PATH = sys.argv[2]
    
    # 2. 执行在线转录
    try:
        # 初始化在线识别器
        transcriber = OnlineAudioTranscriber(
            model_size=MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        
        # 转录音频
        result_text = transcriber.transcribe(AUDIO_PATH)
        
        # 保存结果到文件
        output_txt = os.path.splitext(AUDIO_PATH)[0] + "_online_transcript.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"✅ 转录结果已保存到: {output_txt}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()