using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace WindBot.Game
{
    /// <summary>
    /// Plays with the Kirafan learning AI: every game message goes to a Python process
    /// (kirafan_ai.bridge.server) that answers with the raw response bytes.
    /// Bots use it when their deck is "KirafanAI:&lt;name&gt;" or "KirafanAI:&lt;name&gt;@&lt;level&gt;" (difficulty
    /// 1-4, kirafan_ai.bridge.server LEVELS; see WindBot/KirafanAI/bridge.cfg); the model is
    /// WindBot/KirafanAI/models/&lt;name&gt;.npz, the bot's chat Dialogs/KirafanAI_&lt;level&gt;.json when there is one.
    /// </summary>
    public class KirafanBridge : IDisposable
    {
        public const string Prefix = "KirafanAI:";

        private readonly Process _proc;
        private readonly BinaryWriter _input;
        private readonly BinaryReader _output;

        public static bool IsBridgeDeck(string deck)
        {
            return deck != null && deck.StartsWith(Prefix);
        }

        public static string ModelName(string deck)
        {
            string name = deck.Substring(Prefix.Length);
            int at = name.IndexOf('@');
            return at < 0 ? name : name.Substring(0, at);
        }

        public static int Level(string deck)
        {
            int at = deck.IndexOf('@');
            int level;
            return at >= 0 && int.TryParse(deck.Substring(at + 1), out level) ? level : 0;
        }

        /// <summary>Dialogs/KirafanAI_&lt;level&gt;.json for a bot with a level when it exists, else fallback.</summary>
        public static string DialogName(string deck, string fallback)
        {
            int level = Level(deck);
            string name = "KirafanAI_" + level;
            string path = Path.Combine(Path.Combine(Program.AssetPath, "Dialogs"), name + ".json");
            return level > 0 && File.Exists(path) ? name : fallback;
        }

        /// <summary>
        /// Deck for one duel: a random .ydk from Decks/KirafanAI_&lt;name&gt;/ when that folder has any
        /// (written by kirafan_ai.bridge.register), else Decks/KirafanAI_&lt;name&gt;.ydk.
        /// </summary>
        public static string PickDeck(string name)
        {
            string dir = Path.Combine(Path.Combine(Program.AssetPath, "Decks"), "KirafanAI_" + name);
            if (Directory.Exists(dir))
            {
                string[] decks = Directory.GetFiles(dir, "*.ydk");
                if (decks.Length > 0)
                    return Path.GetFullPath(decks[new Random().Next(decks.Length)]);
            }
            return "KirafanAI_" + name;
        }

        private static Dictionary<string, string> ReadConfig(string path)
        {
            var cfg = new Dictionary<string, string>();
            foreach (string line in File.ReadAllLines(path, Encoding.UTF8))
            {
                int eq = line.IndexOf('=');
                if (eq <= 0 || line.TrimStart().StartsWith("#"))
                    continue;
                cfg[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
            }
            return cfg;
        }

        public KirafanBridge(string name, int level)
        {
            string root = Path.GetFullPath(Path.Combine(Program.AssetPath, "KirafanAI"));
            var cfg = ReadConfig(Path.Combine(root, "bridge.cfg"));
            // paths in bridge.cfg may be relative to WindBot/KirafanAI (portable bundle)
            string python = Path.GetFullPath(Path.Combine(root, cfg["python"]));
            string workdir = Path.GetFullPath(Path.Combine(root, cfg["workdir"]));
            string model = Path.Combine(Path.Combine(root, "models"), name);
            string logDir = Path.Combine(root, "logs");
            Directory.CreateDirectory(logDir);
            string log = Path.Combine(logDir, DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") + "_" + name + ".log");
            string mode = cfg.ContainsKey("mode") ? cfg["mode"] : "best";

            var psi = new ProcessStartInfo(python,
                "-u -m kirafan_ai.bridge.server --model \"" + model + "\" --mode " + mode +
                (level > 0 ? " --level " + level : "") + " --log \"" + log + "\"");
            psi.WorkingDirectory = workdir;
            psi.EnvironmentVariables["KIRAFAN_GAME_ROOT"] = Path.GetFullPath(Path.Combine(root, Path.Combine("..", "..")));
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardInput = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            _proc = Process.Start(psi);
            _proc.ErrorDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                    File.AppendAllText(log, e.Data + Environment.NewLine, Encoding.UTF8);
            };
            _proc.BeginErrorReadLine();
            _input = new BinaryWriter(_proc.StandardInput.BaseStream);
            _output = new BinaryReader(_proc.StandardOutput.BaseStream);
            Logger.WriteLine("KirafanAI bridge started: " + model);
        }

        /// <summary>Sends one game message (message id + body); returns the response, or null.</summary>
        public byte[] Handle(byte[] message)
        {
            _input.Write(message.Length);
            _input.Write(message);
            _input.Flush();
            int length = _output.ReadInt32();
            return length > 0 ? _output.ReadBytes(length) : null;
        }

        public void Dispose()
        {
            try
            {
                _input.Close();
                if (!_proc.WaitForExit(3000))
                    _proc.Kill();
            }
            catch (Exception)
            {
            }
        }
    }
}
