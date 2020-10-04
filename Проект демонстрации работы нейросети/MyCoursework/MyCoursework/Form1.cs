using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ClassifierLibrary;
using System.Security;
using System.Diagnostics.Tracing;

namespace MyCoursework
{
    public partial class Form1 : Form
    {
        private TextBox outputBox;
        private TextBox inputBox;
        private Button recognize;
        private Classifier bot;
        private OpenFileDialog openFileDialog1;
        private Button selectButton;
        private TextBox fileBox;
        private ToolStripContainer toolStripContainer;
        private ToolStrip toolStrip;
        private ToolStripButton aboutButton = new ToolStripButton();
        private ToolStripButton information = new ToolStripButton();

        private void RecognizeClicked(object sender, System.EventArgs e)
        {

            this.outputBox.Text = "Выполнение...";
            string input = this.inputBox.Text;
            bot = new Classifier();
            string answer = bot.GetAnswer(input);
            
            this.outputBox.Text = answer;
        }

        private void SelectClicked(object sender, System.EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string path = null;
                try
                {
                    this.fileBox.Text = openFileDialog1.FileName;
                    path = openFileDialog1.FileName;
                    this.inputBox.Text = File.ReadAllText(path);
                }
                catch (SecurityException ex)
                {
                    MessageBox.Show($"Security error.\n\nError message: {ex.Message}\n\n" +
                    $"Details:\n\n{ex.StackTrace}");
                } catch(Exception ex)
                {
                    MessageBox.Show($"При считывании файла {path} произошла ошибка: " +
                        Environment.NewLine + ex.Message);
                }
            }
        }

        private void AboutClicked(object sender, System.EventArgs e)
        {
            MessageBox.Show("Данное приложение было создано Мелехиным Денисом в 2020 году и" +
                " предназначено для демонстрации работы " +
                "модели LSTM сети и наивного Байесовского класификатора на примере распознования " +
                "информативности текста.");
        }


        private void InformationClicked(object sender, System.EventArgs e)
        {
            MessageBox.Show("Вы можете задать программе входные данные двумя способами:" + Environment.NewLine
                + "1) Ввести текст на английском в поле для входных данных и нажать кнопку \"Проверить\""
                + Environment.NewLine + "2) Выбрать доступный файл с помощью кнопки \"Выбрать файл\""
                + " и нажать на кнопку \"Проверить\""
                + Environment.NewLine + "Результат будет выведен в поле для выходных данных. "
                + Environment.NewLine + "Анализ информативности текста проводился двумя методами: "
                + Environment.NewLine + "1) LSTM сетью"
                + Environment.NewLine + "2) Наивным Байесовским классификатором");
        }
        private void inputBox_Enter(object sender, EventArgs e) //происходит когда элемент стает активным
        {
            this.inputBox.Text = null;
            this.inputBox.ForeColor = Color.Black;
        }

        public void InitializeComponent2()
        {
            this.outputBox = new System.Windows.Forms.TextBox();
            this.inputBox = new System.Windows.Forms.TextBox();
            this.recognize = new Button();
            this.selectButton = new Button();
            this.fileBox = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // outputBox
            // 
            this.outputBox.ReadOnly = true;
            this.outputBox.Multiline = true;
            this.outputBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.outputBox.Location = new Point(650, 120);
            this.outputBox.Size = new System.Drawing.Size(350, 500);
            this.outputBox.Text = "Результат:";
            // 
            // inputBox
            // 
            this.inputBox.ReadOnly = false;
            this.inputBox.Multiline = true;
            this.inputBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.inputBox.Location = new Point(50, 120);
            this.inputBox.Size = new System.Drawing.Size(550, 500);
            this.inputBox.Text = "Введите текст для распознавания его информативности";
            this.inputBox.ForeColor = Color.Gray;
            this.inputBox.Enter += inputBox_Enter;
            // 
            // recognizeButton
            // 
            this.recognize.Text = "Проверить";
            this.recognize.Location = new Point(50, 70);
            this.recognize.Size = new System.Drawing.Size(100, 22);
            this.recognize.Click += RecognizeClicked;

            //
            //fileDialog
            //

            this.openFileDialog1 = new OpenFileDialog();
            this.openFileDialog1.ValidateNames = false;
            this.openFileDialog1.CheckFileExists = true;
            this.openFileDialog1.CheckPathExists = true;
            this.openFileDialog1.Filter = "Text files (*.txt)|*.txt";

            //
            //selectButton
            //
            this.selectButton.Text = "Загрузить из файла";
            this.selectButton.Location = new Point(460, 39);
            this.selectButton.Size = new System.Drawing.Size(140, 22);
            this.selectButton.Click += SelectClicked;

            //
            //fileBox
            //

            this.fileBox.ReadOnly = true;
            this.fileBox.Location = new Point(50, 40);
            this.fileBox.Size = new System.Drawing.Size(390, 45);

            //
            //aboutButton
            //

            this.aboutButton = new ToolStripButton();
            aboutButton.Text = "О программе";
            aboutButton.Size = new System.Drawing.Size(40, 15);
            aboutButton.Click += AboutClicked;

            //
            //informationButton
            //

            this.information = new ToolStripButton();
            information.Text = "Справка";
            information.Size = new System.Drawing.Size(40, 15);
            information.Click += InformationClicked;

            //
            //ToolStrip
            //

            this.toolStripContainer = new System.Windows.Forms.ToolStripContainer();
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            // Add items to the ToolStrip.
            this.toolStrip.Items.Add(aboutButton);
            this.toolStrip.Items.Add(information);
            this.toolStrip.CanOverflow = false;

            // Add the ToolStrip to the top panel of the ToolStripContainer.
            this.toolStripContainer.Size = new System.Drawing.Size(160, 25);
            this.toolStripContainer.TopToolStripPanel.Controls.Add(this.toolStrip);
            // Add the ToolStripContainer to the form.
            Controls.Add(this.toolStripContainer);

            // 
            // Form1
            // 
            this.ClientSize = new System.Drawing.Size(1050, 650);
            this.Controls.Add(this.outputBox);
            this.Controls.Add(this.inputBox);
            this.Controls.Add(this.recognize);
            this.Controls.Add(this.selectButton);
            this.Controls.Add(this.fileBox);
            this.Text = "Coursework";
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        public Form1()
        {
            InitializeComponent2();
        }
    }
}
